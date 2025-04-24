"""
NLP_Project
"""

##############################
# 1. Импорты и начальная настройка
##############################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import joblib  # Используется для сохранения артефактов RNN
from collections import Counter
from itertools import chain
from tqdm import tqdm  # Альтернатива для не-ноутбучных сред

# PyTorch
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

# Hugging Face Transformers
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# Scikit-learn (используется для train_test_split и метрик)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

try:
    from navec import Navec  # Для эмбеддингов RNN
except ImportError:
    print("Предупреждение: Библиотека navec не установлена. Функциональность RNN с Navec будет недоступна.")
    Navec = None
try:
    from googletrans import Translator  # Для перевода (опционально)
except ImportError:
    print("Предупреждение: Библиотека googletrans не установлена. Функция перевода будет недоступна.")
    Translator = None
import asyncio
import traceback

# --- Глобальные константы и конфигурация ---
BASE_DATA_DIR = './dataset/'
EN_DATA_PATH = os.path.join(BASE_DATA_DIR, 'train-balanced-sarcasm.csv')
RU_TRANSLATED_DATA_PATH = os.path.join(BASE_DATA_DIR, 'train_ru.csv')
JOKES_DATA_PATH = os.path.join(BASE_DATA_DIR, 'jokes.csv')
JOKES_CSV_PATH = os.path.join(BASE_DATA_DIR, 'train_ru_jokes.csv')
TRAIN_FINAL_PATH = os.path.join(BASE_DATA_DIR, 'train_final.csv')
TEST_FINAL_PATH = os.path.join(BASE_DATA_DIR, 'test_final.csv')
MODEL_SAVE_DIR = os.path.join(BASE_DATA_DIR, 'models/')
NAVEC_PATH = os.path.join(BASE_DATA_DIR, 'navec_hudlit_v1_12B_500K_300d_100q.tar')

os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {DEVICE}")

navec_model = None


##############################
# 2. Вспомогательные функции
##############################

def plot_confusion_matrix(y_true, y_pred, title='Матрица ошибок'):
    """Отрисовывает матрицу ошибок с использованием seaborn."""
    data = {'y_Actual': y_true, 'y_Predicted': y_pred}
    df_cm = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    conf_matrix = pd.crosstab(df_cm['y_Actual'], df_cm['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plot_filename = os.path.join(MODEL_SAVE_DIR, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(plot_filename)
    print(f"Матрица ошибок сохранена в: {plot_filename}")
    plt.close()


def test_data_analysis(Y_pred, Y_test, model_name="Модель"):
    """Рассчитывает и выводит метрики классификации и матрицу ошибок."""
    precision = accuracy_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred, average='binary', zero_division=0)
    f1 = f1_score(Y_test, Y_pred, average='binary', zero_division=0)
    print(f"\n--- Метрики для {model_name} ---")
    print(f"Accuracy (Точность): {precision:.4f}")
    print(f"Recall (Полнота):   {recall:.4f}")
    print(f"F1_score:          {f1:.4f}")
    print("-" * (20 + len(model_name)))
    plot_confusion_matrix(Y_test, Y_pred, title=f'{model_name} - Матрица ошибок')
    return {"accuracy": precision, "recall": recall, "f1": f1}


##############################
# 3. Подготовка данных
##############################

async def translate_english_data(input_path=EN_DATA_PATH, output_path=RU_TRANSLATED_DATA_PATH, batch_size=50,
                                 delay_between_batches=0.2, num_samples=None, sample_seed=42):
    """
    АСИНХРОННЫЙ перевод английского датасета сарказма на русский.
    Использует asyncio для обработки асинхронных вызовов googletrans.
    ВНИМАНИЕ: зависит от API Google Translate.
    """
    if Translator is None:
        print("Ошибка: Библиотека googletrans не установлена. Пропуск перевода.")
        return

    print("--- Начало АСИНХРОННОГО перевода английского датасета ---")
    print(f"Входной файл: {input_path}")
    print(f"Выходной файл: {output_path}")
    print(f"Количество строк для перевода: {num_samples if num_samples else 'все'}")
    print(f"Random seed: {sample_seed}")
    print(f"Размер батча: {batch_size}, Задержка между батчами: {delay_between_batches} сек")
    print("ВНИМАНИЕ: Этот процесс все еще может занять много времени!")

    try:
        df_en = pd.read_csv(input_path)

        if num_samples is not None:
            df_en = df_en.sample(n=min(num_samples, len(df_en)),
                                 random_state=sample_seed)
            print(f"Выбрано {len(df_en)} случайных строк для перевода")

        comment_col = None
        label_col = None
        text_cols = ['text', 'comment', 'parent_comment']
        label_cols = ['Y', 'label']
        comment_col = next((col for col in text_cols if col in df_en.columns), None)
        label_col = next((col for col in label_cols if col in df_en.columns), None)

        if comment_col is None or label_col is None:
            print(f"Ошибка: Не найдены колонки (текст/метка) в {input_path}.")
            print(f"Найденные колонки: {df_en.columns.tolist()}")
            return
        print(f"Используются колонки: текст='{comment_col}', метка='{label_col}'")

        translator = Translator()
        processed_rows = []
        total_rows = len(df_en)

        for i in tqdm(range(0, total_rows, batch_size), desc="Обработка батчей"):
            batch_df = df_en.iloc[i:min(i + batch_size, total_rows)]
            tasks = []
            original_data_for_batch = []

            for index, row in batch_df.iterrows():
                text_to_translate = row.get(comment_col, '')
                label = row.get(label_col, None)
                if text_to_translate and label is not None and isinstance(text_to_translate, str):
                    coro = translator.translate(text_to_translate, dest='ru')
                    tasks.append(coro)
                    original_data_for_batch.append({'index': index, 'label': label, 'text': text_to_translate})

            if not tasks: continue

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as gather_err:
                print(f"\nКритическая ошибка во время asyncio.gather: {gather_err}")
                continue

            for j, result in enumerate(results):
                original = original_data_for_batch[j]
                if isinstance(result, Exception):
                    print(f"\nОшибка перевода строки {original['index']}: {result}. Текст: {original['text'][:50]}...")
                elif hasattr(result, 'text'):
                    processed_rows.append({'comment': result.text, 'label': original['label']})
                else:
                    print(
                        f"\nПредупреждение: Неожиданный результат для строки {original['index']}: {type(result)}. Результат: {result}")

            if delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)

        if not processed_rows:
            print("\nОшибка: Не удалось перевести ни одной строки.")
            return

        df_ru = pd.DataFrame(processed_rows)
        df_ru['label'] = df_ru['label'].astype(int)

        print(f"\nПеревод завершен ({len(df_ru)} строк успешно). Сохранение в {output_path}...")
        df_ru.to_csv(output_path, index=False)
        print("Файл сохранен.")
        print("--- Асинхронный перевод английского датасета завершен ---")

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {input_path}")
    except Exception as e:
        print(f"Непредвиденная ошибка во время перевода: {e}")
        traceback.print_exc()


def prepare_jokes_csv(input_path=JOKES_DATA_PATH, output_path=JOKES_CSV_PATH):
    """Читает CSV с шутками, обрабатывает и сохраняет в промежуточный файл."""
    print("--- Подготовка CSV файла с шутками ---")
    try:
        df_jokes = pd.read_csv(input_path)
        cols_to_drop = ['theme', 'rating']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df_jokes.columns]
        if existing_cols_to_drop:
            df_jokes = df_jokes.drop(columns=existing_cols_to_drop)
        if 'text' in df_jokes.columns:
            df_jokes = df_jokes.rename(columns={"text": "comment"})
        elif 'comment' not in df_jokes.columns:
            print("Ошибка: Не найдена колонка 'comment' или 'text' в файле шуток.")
            return
        df_jokes['label'] = 1
        df_jokes.dropna(subset=['comment'], inplace=True)
        df_jokes = df_jokes[df_jokes['comment'].astype(str).str.strip().astype(bool)]
        df_jokes = df_jokes[['comment', 'label']]
        print(f"Обработано {len(df_jokes)} шуток. Сохранение в {output_path}...")
        df_jokes.to_csv(output_path, index=False)
        print("Файл сохранен.")
        print("--- Подготовка CSV с шутками завершена ---")
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {input_path}")
    except Exception as e:
        print(f"Ошибка при обработке файла шуток: {e}")


def create_final_dataset(translated_path=RU_TRANSLATED_DATA_PATH,
                         jokes_path=JOKES_CSV_PATH,
                         train_output_path=TRAIN_FINAL_PATH,
                         test_output_path=TEST_FINAL_PATH,
                         non_sarcasm_num=20000, sarcasm_num=10000, jokes_num=10000,
                         train_split_ratio=0.7, random_seed=42):
    """
    Загружает переведенные комментарии и шутки, объединяет, очищает, разделяет и сохраняет.
    """
    print("--- Формирование итогового датасета ---")
    np.random.seed(random_seed)
    try:
        print(f"Загрузка переведенных комментариев из: {translated_path}")
        if not os.path.exists(translated_path):
            raise FileNotFoundError(f"Файл не найден: {translated_path}")
        df_en_sarcasm_or_not = pd.read_csv(translated_path)
        if 'comment' not in df_en_sarcasm_or_not.columns or 'label' not in df_en_sarcasm_or_not.columns:
            raise ValueError(
                "Нет колонок comment/label")

        print(f"Загрузка подготовленных шуток из: {jokes_path}")
        if not os.path.exists(jokes_path): raise FileNotFoundError(f"Файл не найден: {jokes_path}")
        df_ru_jokes = pd.read_csv(jokes_path)
        if 'comment' not in df_ru_jokes.columns or 'label' not in df_ru_jokes.columns:
            raise ValueError(
                "Нет колонок comment/label")

    except (FileNotFoundError, ValueError) as e:
        print(f"Ошибка: {e}")
        return
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    df_en_non_sarcasm = df_en_sarcasm_or_not[df_en_sarcasm_or_not['label'] == 0]
    df_en_sarcasm = df_en_sarcasm_or_not[df_en_sarcasm_or_not['label'] == 1]

    print(
        f"Доступно НЕ саркастичных: {len(df_en_non_sarcasm)}, саркастичных: {len(df_en_sarcasm)}, шуток: {len(df_ru_jokes)}")

    df_non_sarcasm_sample = df_en_non_sarcasm.sample(n=min(non_sarcasm_num, len(df_en_non_sarcasm)),
                                                     random_state=random_seed)
    df_sarcasm_sample = df_en_sarcasm.sample(n=min(sarcasm_num, len(df_en_sarcasm)), random_state=random_seed)
    df_jokes_sample = df_ru_jokes.sample(n=min(jokes_num, len(df_ru_jokes)), random_state=random_seed)

    print(
        f"Выбрано НЕ саркастичных: {len(df_non_sarcasm_sample)}, саркастичных: {len(df_sarcasm_sample)}, шуток: {len(df_jokes_sample)}")

    df_final = pd.concat([df_non_sarcasm_sample, df_sarcasm_sample, df_jokes_sample]).sample(frac=1,
                                                                                             random_state=random_seed).reset_index(
        drop=True)
    print(f"Общий размер до очистки: {len(df_final)}")

    print("Очистка текста...")
    df_final['comment'] = df_final['comment'].astype(str)
    df_final['comment'] = df_final['comment'].replace({r'\r': '', r'\n': ''}, regex=True)
    df_final['comment'] = df_final['comment'].str.replace(r'[^\w\s]', '', regex=True)
    df_final['comment'] = df_final['comment'].str.lower()
    df_final['comment'] = df_final['comment'].str.strip()
    original_len = len(df_final)
    df_final.dropna(subset=['comment'], inplace=True)
    df_final = df_final[df_final['comment'] != '']
    print(f"Удалено {original_len - len(df_final)} пустых строк. Размер после очистки: {len(df_final)}")

    print("\nРаспределение меток в итоговом датасете:")
    print(df_final['label'].value_counts(normalize=True))
    sns.set_style("darkgrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df_final.label)
    plt.title("Распределение меток")
    plot_filename = os.path.join(MODEL_SAVE_DIR, "final_dataset_label_distribution.png")
    plt.savefig(plot_filename)
    print(f"График сохранен: {plot_filename}")
    plt.close()

    print(f"Разделение на train/test ({train_split_ratio * 100:.0f}%/{100 - train_split_ratio * 100:.0f}%)...")
    df_train, df_test = train_test_split(df_final, train_size=train_split_ratio, random_state=random_seed,
                                         stratify=df_final['label'])
    print(f"Обучающая выборка: {len(df_train)}, Тестовая выборка: {len(df_test)}")

    print(f"Сохранение обучающей выборки в: {train_output_path}")
    df_train.to_csv(train_output_path, index=False)
    print(f"Сохранение тестовой выборки в: {test_output_path}")
    df_test.to_csv(test_output_path, index=False)
    print("--- Формирование итогового датасета завершено ---")


##############################
# 4. Transformers (BERT)
##############################

class CustomDatasetBERT(TorchDataset):
    def __init__(self, texts, targets, tokenizer, max_len=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if isinstance(self.texts, pd.Series) else self.texts[idx])
        target = int(self.targets.iloc[idx] if isinstance(self.targets, pd.Series) else self.targets[idx])
        text = text[:self.max_len]
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                              return_token_type_ids=False, padding='max_length', truncation=True,
                                              return_attention_mask=True, return_tensors='pt')
        return {'text': text, 'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.long)}


class BertClassifier:
    def __init__(self, model_path, tokenizer_path, n_classes=2, epochs=1, batch_size=2,
                 model_save_path='/content/bert.pt'):
        print(f"Инициализация BertClassifier: model={model_path}")
        try:
            # Указываем ignore_mismatched_sizes=True, чтобы игнорировать несовпадение размера классификатора
            self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=n_classes,
                                                                       ignore_mismatched_sizes=True)
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки BERT из '{model_path}': {e}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"BertClassifier использует: {self.device}")
        self.model_save_path = model_save_path
        self.max_len = 512
        self.epochs = epochs
        self.batch_size = batch_size

        self.model.to(self.device)
        self.train_loader, self.valid_loader, self.optimizer, self.scheduler, self.loss_fn = None, None, None, None, None

    def preparation(self, X_train, y_train, X_valid, y_valid):
        print("Подготовка данных для BERT...")
        self.train_set = CustomDatasetBERT(X_train, y_train, self.tokenizer, self.max_len)
        self.valid_set = CustomDatasetBERT(X_valid, y_valid, self.tokenizer, self.max_len)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        num_training_steps = len(self.train_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=num_training_steps)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        print("Подготовка данных завершена.")

    def fit(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        loader = tqdm(self.train_loader, desc=f"Обучение BERT", leave=False)
        for data in loader:
            ids, mask, targets = data["input_ids"].to(self.device), data["attention_mask"].to(self.device), data[
                "targets"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(input_ids=ids, attention_mask=mask).logits
            preds = torch.argmax(logits, dim=1)
            loss = self.loss_fn(logits, targets)
            correct_predictions += torch.sum(preds == targets)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            loader.set_postfix(loss=loss.item())
        return (correct_predictions.double() / len(self.train_set)).item(), total_loss / len(self.train_loader)

    def eval(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        loader = tqdm(self.valid_loader, desc="Валидация BERT", leave=False)
        with torch.no_grad():
            for data in loader:
                ids, mask, targets = data["input_ids"].to(self.device), data["attention_mask"].to(self.device), data[
                    "targets"].to(self.device)
                logits = self.model(input_ids=ids, attention_mask=mask).logits
                preds = torch.argmax(logits, dim=1)
                loss = self.loss_fn(logits, targets)
                correct_predictions += torch.sum(preds == targets)
                total_loss += loss.item()
                loader.set_postfix(loss=loss.item())
        return (correct_predictions.double() / len(self.valid_set)).item(), total_loss / len(self.valid_loader)

    def train(self):
        best_accuracy = 0
        print(f"\n--- Обучение BERT ({self.epochs} эпох) ---")
        for epoch in range(self.epochs):
            print(f"\nЭпоха {epoch + 1}/{self.epochs}")
            train_acc, train_loss = self.fit()
            print(f'Обучение: Loss={train_loss:.4f} Acc={train_acc:.4f}')
            val_acc, val_loss = self.eval()
            print(f'Валидация: Loss={val_loss:.4f} Acc={val_acc:.4f}')
            print('-' * 20)
            if val_acc > best_accuracy:
                print(f"Лучшая модель! Acc: {val_acc:.4f}. Сохранение: {self.model_save_path}")
                torch.save(self.model.state_dict(), self.model_save_path)
                best_accuracy = val_acc
        print(f"--- Обучение BERT завершено. Лучшая Acc: {best_accuracy:.4f} ---")
        print(f"Загрузка лучшей модели из {self.model_save_path}")
        if os.path.exists(self.model_save_path):
            try:
                self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
                self.model.to(
                    self.device)
            except Exception as e:
                print(f"Предупреждение: Ошибка загрузки лучшей BERT модели: {e}.")
        else:
            print("Предупреждение: Файл лучшей BERT модели не найден.")

    def predict(self, text):
        """Предсказание метки и вероятностей для одного текста."""
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text[:self.max_len], add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        ids = encoding['input_ids'].to(self.device)
        mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()  # Получаем вероятности
            prediction = torch.argmax(logits, dim=1).cpu().item()

        # Возвращаем и предсказание, и вероятности
        return prediction, probabilities

    def evaluate_on_test_set(self, X_test, Y_test):
        print("\n--- Оценка BERT на тесте ---")
        self.model.eval()
        y_pred = []
        X_test_list = X_test.tolist() if isinstance(X_test, pd.Series) else list(X_test)
        Y_test_list = Y_test.tolist() if isinstance(Y_test, pd.Series) else list(Y_test)
        for comment in tqdm(X_test_list, desc="Предсказание BERT на тесте"):
            if isinstance(comment, str):
                pred_label, _ = self.predict(comment)
                y_pred.append(pred_label)
            else:
                print(f"Предупреждение: Пропуск нестроки в тесте: {comment}")
                y_pred.append(-1)
        valid_indices = [i for i, p in enumerate(y_pred) if p != -1]
        filtered_y_pred = [y_pred[i] for i in valid_indices]
        filtered_Y_test = [Y_test_list[i] for i in valid_indices]
        if not filtered_Y_test:
            print("Ошибка: Нет валидных предсказаний на тесте.")
            return
        test_data_analysis(filtered_y_pred, filtered_Y_test, model_name="BERT")
        print("--- Оценка BERT на тесте завершена ---")


def run_train_transformer(train_csv_path=TRAIN_FINAL_PATH, test_csv_path=TEST_FINAL_PATH,
                          model_name='cointegrated/rubert-tiny2', epochs=10, batch_size=64,
                          val_split_ratio=0.15, model_save_path=os.path.join(MODEL_SAVE_DIR, 'bert_classifier.pt'),
                          random_seed=42):
    """ Запускает обучение и оценку модели BERT. """
    print("--- Запуск обучения Transformer (BERT) ---")
    try:
        print(f"Загрузка данных: {train_csv_path}, {test_csv_path}")
        if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
            raise FileNotFoundError("Нет train/test")
        df_train_full, df_test = pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)
        for df in [df_train_full, df_test]:
            df.dropna(subset=['comment', 'label'], inplace=True)
            df['label'] = df['label'].astype(int)
            df['comment'] = df['comment'].astype(str)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    print(f"Разделение train/validation ({1 - val_split_ratio:.0%}/{val_split_ratio:.0%})...")
    X, Y = df_train_full['comment'], df_train_full['label']
    train_texts, val_texts, train_labels, val_labels = train_test_split(X, Y, test_size=val_split_ratio,
                                                                        random_state=random_seed, stratify=Y)

    classifier = BertClassifier(model_path=model_name, tokenizer_path=model_name, n_classes=2,
                                epochs=epochs, batch_size=batch_size, model_save_path=model_save_path)
    classifier.preparation(X_train=train_texts, y_train=train_labels, X_valid=val_texts, y_valid=val_labels)
    classifier.train()
    classifier.evaluate_on_test_set(df_test['comment'], df_test['label'])
    print("--- Обучение Transformer (BERT) завершено ---")


def run_predict_transformer(text, model_name='cointegrated/rubert-tiny2',
                            model_load_path=os.path.join(MODEL_SAVE_DIR, 'bert_classifier.pt')):
    """ Запускает предсказание BERT для текста и выводит вероятности. """  # <-- Изменено описание
    print("--- Запуск предсказания Transformer (BERT) ---")
    if not os.path.exists(model_load_path):
        print(f"Ошибка: Модель BERT не найдена: {model_load_path}")
        return None
    try:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Ошибка загрузки BERT: {e}")
        return None

    # Создаем временный объект классификатора для вызова predict
    temp_classifier = BertClassifier(model_path=model_name, tokenizer_path=model_name, model_save_path=model_load_path)
    temp_classifier.model = model
    temp_classifier.tokenizer = tokenizer
    temp_classifier.device = device

    # Получаем и метку, и вероятности
    prediction, probabilities = temp_classifier.predict(text)

    label_map = {0: 'Не сарказм/шутка', 1: 'Сарказм/шутка'}
    print(f"\nИсходный текст: '{text}'")
    print(f"Предсказание (BERT): {label_map.get(prediction, '?')} (Метка: {prediction})")
    # Выводим вероятности
    print(f"Вероятности [0, 1]: [{probabilities[0]:.4f}, {probabilities[1]:.4f}]")
    print("--- Предсказание Transformer (BERT) завершено ---")
    return prediction


##############################
# 5. RNN (Рекуррентная нейронная сеть)
##############################

RNN_MAX_VOCAB_SIZE = 30000


class TokenizerRNN:
    def __init__(self, word_pattern="[\w']+"): self.word_pattern = re.compile(word_pattern)

    def tokenize(self, text): return self.word_pattern.findall(str(text).lower()) if isinstance(text,
                                                                                                (str, np.str_)) else []


class VocabRNN:
    def __init__(self, tokenized_texts, max_vocab_size=None):
        print("Построение словаря RNN...")
        counts = Counter(chain(*tokenized_texts))
        max_vocab_size = max_vocab_size or len(counts)
        common_pairs = counts.most_common(max_vocab_size - 3)
        self.PAD_TOKEN, self.UNK_TOKEN, self.EOS_TOKEN = "<PAD>", "<UNK>", "<EOS>"
        self.PAD_IDX, self.UNK_IDX, self.EOS_IDX = 0, 1, 2
        self.itos = [self.PAD_TOKEN, self.UNK_TOKEN, self.EOS_TOKEN] + [p[0] for p in common_pairs]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        print(f"Размер словаря RNN: {len(self.itos)}")

    def vectorize(self, t_list): return [self.stoi.get(t, self.UNK_IDX) for t in t_list] + [self.EOS_IDX]

    def __len__(self): return len(self.itos)


def prepare_emb_matrix(navec_model: Navec, vocab: VocabRNN):
    if navec_model is None:
        print("Ошибка: Navec не передан.")
        return None
    if not isinstance(navec_model, Navec):
        print("Ошибка: Не Navec модель.")
        return None
    print("Подготовка эмбеддингов RNN из Navec...")
    embedding_dim = 300
    try:
        sample_word = next(iter(navec_model.vocab.words))
        embedding_dim = len(navec_model[sample_word])
        print(f"Размерность Navec: {embedding_dim}")
    except Exception as e:
        print(f"Предупреждение: Ошибка определения размерности Navec ({e}). Используется {embedding_dim}.")
    emb_matrix = torch.zeros((len(vocab), embedding_dim), dtype=torch.float32)
    all_vecs = [navec_model[w] for w in vocab.itos if
                w not in [vocab.PAD_TOKEN, vocab.UNK_TOKEN, vocab.EOS_TOKEN] and w in navec_model]
    if all_vecs:
        mean, std = np.mean(all_vecs, 0), np.std(all_vecs, 0)
        std[std == 0] = 1e-6
    else:
        print("Предупреждение: Нет слов из словаря в Navec. OOV ~ N(0,1).")
        mean, std = np.zeros(embedding_dim), np.ones(
            embedding_dim)
    init_count = 0
    for w, idx in vocab.stoi.items():
        if w == vocab.PAD_TOKEN:
            emb_matrix[idx] = torch.zeros(embedding_dim)
        elif w == vocab.UNK_TOKEN:
            emb_matrix[idx] = torch.tensor(mean, dtype=torch.float32)
        elif w == vocab.EOS_TOKEN:
            emb_matrix[idx] = torch.zeros(embedding_dim)
        elif w in navec_model:
            emb_matrix[idx] = torch.tensor(navec_model[w], dtype=torch.float32)
            init_count += 1
        else:
            emb_matrix[idx] = torch.tensor(np.random.normal(mean, std), dtype=torch.float32)
    print(f"Инициализация {init_count}/{len(vocab) - 3} векторов из Navec. Матрица: {emb_matrix.shape}")
    return emb_matrix


class RecurrentClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, bidirectional, pad_idx,
                 emb_matrix=None):
        super().__init__();
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if emb_matrix is not None:
            print("Загрузка эмбеддингов в RNN...")
            self.embedding.weight.data.copy_(emb_matrix)
            self.embedding.weight.requires_grad = False
            print("Эмбеддинги заморожены.")
        else:
            print("Предупреждение: Эмбеддинги не предоставлены.")
            self.embedding.weight.requires_grad = True
        self.rnn = torch.nn.GRU(embedding_dim, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0,
                                bidirectional=bidirectional, batch_first=True)
        lin_in_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout_out = torch.nn.Dropout(dropout)
        fc_hidden = 40
        self.out_proj = torch.nn.Sequential(torch.nn.Linear(lin_in_size, fc_hidden), torch.nn.ReLU(),
                                            self.dropout_out, torch.nn.Linear(fc_hidden, 2))

    def forward(self, packed_input):
        emb_data = self.dropout_out(self.embedding(packed_input.data))  # Dropout на эмбеддингах
        packed_emb = torch.nn.utils.rnn.PackedSequence(emb_data, packed_input.batch_sizes, packed_input.sorted_indices,
                                                       packed_input.unsorted_indices)
        _, hidden = self.rnn(packed_emb)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        return self.out_proj(hidden)  # Dropout уже внутри


class TrainerRNN:
    def __init__(self, config):
        self.cfg = config
        self.epochs = config.get("n_epochs", 10)
        self.lr = config.get("lr", 1e-3)
        self.wd = config.get("weight_decay", 1e-6)
        self.dev = config.get("device", "cpu")
        self.save_path = config.get("model_save_path", None)
        self.verb = config.get("verbose", True)
        self.model, self.opt, self.loss_fn = None, None, torch.nn.CrossEntropyLoss().to(self.dev)
        self.hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    def setup_opt(self, m):
        return torch.optim.Adam(m.parameters(), lr=self.lr, weight_decay=self.wd)

    def fit(self, m, tr_ldr, v_ldr):
        self.model = m.to(self.dev)
        self.opt = self.setup_opt(self.model)
        best_f1 = -1.0
        print(f"\n--- Обучение RNN ({self.epochs} эпох) ---")
        for ep in range(self.epochs):
            print(f"\nЭпоха {ep + 1}/{self.epochs}")
            tr_loss = self._train_ep(tr_ldr)
            v_info = self._val_ep(v_ldr)
            self.hist["train_loss"].append(tr_loss)
            self.hist["val_loss"].append(v_info["loss"])
            self.hist["val_acc"].append(v_info["acc"])
            self.hist["val_f1"].append(v_info["f1"])
            if self.verb: print(
                f"Итог: Train Loss={tr_loss:.4f}, Val Loss={v_info['loss']:.4f}, Acc:{v_info['acc']:.4f}, F1:{v_info['f1']:.4f}")
            if v_info["f1"] > best_f1:
                best_f1 = v_info["f1"]
            if self.save_path:
                print(f"Лучший F1 -> {best_f1:.4f}. Сохранение: {self.save_path}");
                torch.save(self.model.state_dict(),
                           self.save_path)
            else:
                print(f"Лучший F1 -> {best_f1:.4f}.")
        print("--- Обучение RNN завершено ---")
        if self.save_path and os.path.exists(self.save_path):
            print(f"Загрузка лучшей RNN модели: {self.save_path}")
            try:
                self.model.load_state_dict(torch.load(self.save_path, map_location=self.dev));
                self.model.to(self.dev)
            except Exception as e:
                print(f"Ошибка загрузки RNN: {e}.")
        return self.model.eval()

    def _train_ep(self, ldr):
        self.model.train()
        tot_loss = 0
        n_b = len(ldr) if ldr else 0
        if n_b == 0:
            return 0.0
        pbar = tqdm(ldr, "Обучение RNN", leave=False) if self.verb else ldr
        for batch in pbar:
            if batch is None or batch[0] is None:
                continue  # Пропуск батчей с ошибкой collate_fn
            self.opt.zero_grad()
            p_txt, lbls = batch

            # Перемещаем и упакованную последовательность, и метки
            # Для PackedSequence нужно переместить его внутренний тензор .data
            # Создаем новый PackedSequence с данными на нужном устройстве
            p_txt = p_txt._replace(data=p_txt.data.to(self.dev))
            lbls = lbls.to(self.dev)  # Перемещаем метки

            logits = self.model(p_txt)  # Передаем данные с GPU/CPU
            loss = self.loss_fn(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            tot_loss += loss.item()
            if self.verb:
                pbar.set_postfix(loss=loss.item())
        return tot_loss / n_b

    def _val_ep(self, ldr):
        self.model.eval()
        tot_loss = 0
        all_p, all_l = [], []
        n_b = len(ldr) if ldr else 0
        if n_b == 0:
            return {"loss": 0.0, "acc": 0.0, "f1": 0.0}
        pbar = tqdm(ldr, "Валидация RNN", leave=False) if self.verb else ldr
        with torch.no_grad():
            for batch in pbar:
                if batch is None or batch[0] is None:
                    continue
                p_txt, lbls = batch
                p_txt = p_txt._replace(data=p_txt.data.to(self.dev))
                lbls = lbls.to(self.dev)

                logits = self.model(p_txt)
                loss = self.loss_fn(logits, lbls)
                tot_loss += loss.item()
                preds = torch.argmax(logits, 1)
                all_p.extend(preds.cpu().tolist())
                all_l.extend(lbls.cpu().tolist())  # Собираем на CPU
                if self.verb:
                    pbar.set_postfix(loss=loss.item())
        avg_loss = tot_loss / n_b
        acc = accuracy_score(all_l, all_p)
        f1 = f1_score(all_l, all_p, average='binary', pos_label=1, zero_division=0)
        return {"loss": avg_loss, "acc": acc, "f1": f1}

    def predict(self, ldr):
        if self.model is None: raise RuntimeError("RNN не обучена.")
        self.model.eval()
        preds = []
        pbar = tqdm(ldr, "Предсказание RNN", leave=False) if self.verb else ldr
        with torch.no_grad():
            for batch in pbar:
                if batch is None or batch[0] is None:
                    continue
                p_txt, _ = batch

                p_txt = p_txt._replace(data=p_txt.data.to(self.dev))

                logits = self.model(p_txt)
                preds.extend(torch.argmax(logits, 1).cpu().tolist())
        return np.asarray(preds)


class TextDatasetRNN(TorchDataset):
    def __init__(self, tokenized_texts, labels, vocab: VocabRNN):
        self.texts = tokenized_texts
        self.labels = labels
        self.vocab = vocab

    def __getitem__(self, idx):
        vec_txt = self.vocab.vectorize(self.texts[idx])
        lbl = self.labels[idx]
        return torch.tensor(vec_txt, dtype=torch.long), torch.tensor(lbl, dtype=torch.long)

    def __len__(self): return len(self.texts)


def collate_fn_rnn(batch):
    texts = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = [t.to(device) for t in texts]
    labels = labels.to(device)

    try:
        packed_texts = torch.nn.utils.rnn.pack_sequence(texts, enforce_sorted=False)
    except RuntimeError as e:
        print(f"Ошибка упаковки: {e}")
        return None, None
    return packed_texts, labels


def run_train_rnn(train_csv_path=TRAIN_FINAL_PATH, test_csv_path=TEST_FINAL_PATH, navec_path=NAVEC_PATH,
                  max_vocab_size=RNN_MAX_VOCAB_SIZE, embedding_dim=300, hidden_size=100, num_layers=3,
                  dropout=0.1, bidirectional=True, epochs=10, batch_size=64, lr=1e-3, weight_decay=1e-6,
                  val_split_ratio=0.2, model_save_path=os.path.join(MODEL_SAVE_DIR, 'rnn_model.pt'),
                  vocab_save_path=os.path.join(MODEL_SAVE_DIR, 'rnn_vocab.joblib'),
                  tokenizer_save_path=os.path.join(MODEL_SAVE_DIR, 'rnn_tokenizer.joblib'), random_seed=42):
    """ Запускает обучение и оценку RNN модели. """
    print("--- Запуск обучения RNN ---")
    global navec_model
    if Navec is None:
        print("Ошибка: Navec не импортирован.")
        return
    try:
        print(f"Загрузка данных: {train_csv_path}, {test_csv_path}")
        if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path): raise FileNotFoundError(
            "Нет train/test")
        df_train_full, df_test = pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)
        for df in [df_train_full, df_test]:
            df.dropna(subset=['comment', 'label'], inplace=True)
            df['label'] = df['label'].astype(int)
            df['comment'] = df['comment'].astype(str)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    print("Токенизация RNN...")
    rnn_tokenizer = TokenizerRNN()
    df_train_full['tokens'] = df_train_full['comment'].apply(rnn_tokenizer.tokenize)
    df_test['tokens'] = df_test['comment'].apply(rnn_tokenizer.tokenize)
    df_train_full = df_train_full[df_train_full['tokens'].map(len) > 0]
    df_test = df_test[df_test['tokens'].map(len) > 0]
    print(f"Размер train после удаления пустых токенов: {len(df_train_full)}")
    print(f"Размер test после удаления пустых токенов: {len(df_test)}")

    vocab = VocabRNN(df_train_full['tokens'].tolist(), max_vocab_size=max_vocab_size)

    if navec_model is None:
        try:
            print(f"Загрузка Navec: {navec_path}")
            if not os.path.exists(navec_path): raise FileNotFoundError(f"Navec не найден: {navec_path}")
            navec_model = Navec.load(navec_path)
            print("Navec загружен.")
            try:
                sample_word = next(iter(navec_model.vocab.words))
                model_dim = len(navec_model[sample_word])
                if model_dim != embedding_dim: print(
                    f"Предупр: Navec dim ({model_dim}) != {embedding_dim}. Исп. {model_dim}")
                embedding_dim = model_dim
            except Exception as e:
                print(f"Предупр: Не опр. Navec dim ({e}). Исп. {embedding_dim}.")
        except Exception as e:
            print(f"Ошибка загрузки Navec: {e}")
            return

    emb_matrix = prepare_emb_matrix(navec_model, vocab)
    if emb_matrix is None:
        print("Ошибка создания эмбеддингов.")
        return

    print(f"Разделение train/validation ({1 - val_split_ratio:.0%}/{val_split_ratio:.0%})...")
    train_texts_tok, val_texts_tok, train_labels, val_labels = train_test_split(
        df_train_full['tokens'].tolist(), df_train_full['label'].tolist(),
        test_size=val_split_ratio, random_state=random_seed, stratify=df_train_full['label'].tolist())

    print("Создание датасетов/загрузчиков RNN...")
    train_ds = TextDatasetRNN(train_texts_tok, train_labels, vocab)
    val_ds = TextDatasetRNN(val_texts_tok, val_labels, vocab)
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_rnn)
    val_ldr = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn_rnn)

    print("Инициализация RNN...")
    rnn_model = RecurrentClassifier(
        len(vocab), embedding_dim, hidden_size, num_layers, dropout, bidirectional, vocab.PAD_IDX, emb_matrix)

    trainer_cfg = {"lr": lr, "n_epochs": epochs, "weight_decay": weight_decay, "device": DEVICE,
                   "model_save_path": model_save_path, "verbose": True}
    trainer = TrainerRNN(trainer_cfg)
    trained_model = trainer.fit(rnn_model, train_ldr, val_ldr)

    print("\n--- Финальная оценка RNN на тесте ---")
    test_texts_tok = df_test['tokens'].tolist()
    test_labels = df_test['label'].tolist()
    test_ds = TextDatasetRNN(test_texts_tok, test_labels, vocab)
    test_ldr = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn_rnn)
    test_preds = trainer.predict(test_ldr)
    test_data_analysis(test_preds, test_labels, model_name="RNN (финал)")

    print(f"Сохранение словаря: {vocab_save_path}")
    joblib.dump(vocab, vocab_save_path)
    print(f"Сохранение токенизатора: {tokenizer_save_path}")
    joblib.dump(rnn_tokenizer, tokenizer_save_path)
    print("--- Обучение RNN завершено ---")


def run_predict_rnn(text, model_load_path=os.path.join(MODEL_SAVE_DIR, 'rnn_model.pt'),
                    vocab_load_path=os.path.join(MODEL_SAVE_DIR, 'rnn_vocab.joblib'),
                    tokenizer_load_path=os.path.join(MODEL_SAVE_DIR, 'rnn_tokenizer.joblib'), navec_path=NAVEC_PATH):
    """ Запускает предсказание RNN для текста. """
    print("--- Запуск предсказания RNN ---")
    global navec_model
    if Navec is None:
        print("Ошибка: Navec не импортирован.")
        return None
    try:
        print(f"Загрузка: {vocab_load_path}")
        vocab = joblib.load(vocab_load_path)
        print(f"Загрузка: {tokenizer_load_path}")
        rnn_tokenizer = joblib.load(tokenizer_load_path)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}. Запустите 'run_train_rnn'.")
        return None
    except Exception as e:
        print(f"Ошибка загрузки артефактов: {e}")
        return None

    hidden_size = 100
    num_layers = 3
    dropout = 0.1
    bidirectional = True
    embedding_dim = 300
    if navec_model is None:
        try:
            print(f"Загрузка Navec: {navec_path}")
            if not os.path.exists(navec_path):
                raise FileNotFoundError(f"Navec не найден: {navec_path}")
            navec_model = Navec.load(navec_path)
            print("Navec загружен.")
        except Exception as e:
            print(f"Предупр: Ошибка Navec ({e}). Исп. emb_dim={embedding_dim}.")
    if navec_model:
        try:
            sample_word = next(iter(navec_model.vocab.words))
            model_dim = len(navec_model[sample_word])
            embedding_dim = model_dim
            print(f"Размерность эмб. для предсказания: {embedding_dim}")
        except Exception as e:
            print(f"Предупр: Не опр. Navec dim ({e}). Исп. {embedding_dim}.")

    try:
        print("Инициализация структуры RNN...")
        model = RecurrentClassifier(len(vocab), embedding_dim, hidden_size, num_layers, dropout, bidirectional,
                                    vocab.PAD_IDX)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Загрузка весов RNN: {model_load_path}")
        if not os.path.exists(model_load_path): raise FileNotFoundError(f"Модель не найдена: {model_load_path}")
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Ошибка инициализации/загрузки RNN: {e}")
        return None

    print(f"\nИсходный текст: '{text}'")
    tokenized = rnn_tokenizer.tokenize(text)
    if not tokenized:
        print("Предупр: Текст пуст после токенизации.")
        return None
    print(f"Токены: {tokenized}")
    vectorized = vocab.vectorize(tokenized)
    print(f"Вектор: {vectorized}")

    device = next(model.parameters()).device

    text_tensor = torch.tensor(vectorized, dtype=torch.long, device=device).unsqueeze(0)

    try:

        packed_input = torch.nn.utils.rnn.pack_sequence([text_tensor.squeeze(0)], enforce_sorted=False)

    except RuntimeError as e:
        print(f"Ошибка упаковки: {e}")
        return None

    with torch.no_grad():
        logits = model(packed_input)
        probs = torch.softmax(logits, 1).squeeze().cpu().numpy()
        pred = torch.argmax(logits, 1).item()
    label_map = {0: 'Не сарказм/шутка', 1: 'Сарказм/шутка'}
    print(f"\nПредсказание (RNN): {label_map.get(pred, '?')} (Метка: {pred})")
    print(f"Вероятности [0, 1]: [{probs[0]:.4f}, {probs[1]:.4f}]")
    print("--- Предсказание RNN завершено ---")
    return pred


##############################
# 6. Точка входа и выбор действия
##############################

if __name__ == "__main__":

    actions = {
        "1": (translate_english_data, "Перевести английский датасет (ОЧЕНЬ ДОЛГО!)"),
        "2": (prepare_jokes_csv, "Подготовить CSV файл с шутками"),
        "3": (create_final_dataset, "Создать финальный датасет (train/test)"),
        "4": (run_train_transformer, "Обучить Transformer (BERT)"),
        "5": (run_train_rnn, "Обучить RNN"),
        "6": (run_predict_transformer, "Предсказать с помощью Transformer (BERT)"),
        "7": (run_predict_rnn, "Предсказать с помощью RNN"),
    }

    while True:
        print("\n" + "=" * 50)
        print("Доступные действия:")
        for key, (_, description) in actions.items(): print(f"  {key}: {description}")
        print("  0: Выход")
        print("=" * 50)
        choice = input("Введите номер действия: ").strip()

        if choice == '0': print("Выход из программы."); break
        selected_action = actions.get(choice)

        if selected_action:
            function_to_call, description = selected_action
            print(f"\n--- Запуск действия: {description} ---")
            try:
                if function_to_call in [run_predict_transformer, run_predict_rnn]:
                    text_to_predict = input("Введите текст для предсказания: ")
                    if not text_to_predict: print("Текст не введен. Действие отменено."); continue
                    if function_to_call == run_predict_rnn:
                        if os.path.exists(NAVEC_PATH):
                            function_to_call(text_to_predict, navec_path=NAVEC_PATH)
                        else:
                            print(f"Ошибка: Navec не найден '{NAVEC_PATH}'.")
                    else:
                        function_to_call(text_to_predict)
                elif function_to_call == translate_english_data:
                    try:
                        num_samples_str = input("Кол-во строк для перевода (пусто = все): ").strip()
                        num_samples = int(num_samples_str) if num_samples_str else None
                        asyncio.run(function_to_call(num_samples=num_samples))
                    except ValueError:
                        print("Ошибка: Введите целое число.")
                    except RuntimeError as e:
                        print(f"Ошибка asyncio: {e}.")
                        print("Попытка запуска в существующем цикле...")
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                print("Не удалось запустить в существующем цикле.")
                            else:
                                loop.run_until_complete(function_to_call(num_samples=num_samples))
                        except Exception as async_err:
                            print(f"Повторная ошибка asyncio: {async_err}")

                elif function_to_call in [create_final_dataset, run_train_transformer, run_train_rnn]:
                    print("Используются параметры по умолчанию.")
                    if function_to_call == run_train_rnn:
                        if os.path.exists(NAVEC_PATH):
                            function_to_call(navec_path=NAVEC_PATH)
                        else:
                            print(f"Ошибка: Navec не найден '{NAVEC_PATH}'.")
                    else:
                        function_to_call()
                else:
                    function_to_call()
                print(f"--- Действие '{description}' завершено ---")
            except Exception as e:
                print(f"\n!!! Ошибка во время выполнения '{description}': {e} !!!")
                traceback.print_exc()
        else:
            print("Неверный выбор.")
    print("\nРабота скрипта завершена.")

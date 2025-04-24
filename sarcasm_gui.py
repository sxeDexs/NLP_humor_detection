import threading
import main
import sys
# Поддержка unpickle RNN: обеспечиваем наличие VocabRNN и TokenizerRNN в __main__
sys.modules['__main__'].__dict__['VocabRNN'] = main.VocabRNN
sys.modules['__main__'].__dict__['TokenizerRNN'] = main.TokenizerRNN
import threading
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import nltk
from pathlib import Path

# Попытка использовать ttkbootstrap для современной темы
try:
    import ttkbootstrap as tb
    BaseApp = tb.Window
    STYLE = {"themename": "flatly"}
except ImportError:
    BaseApp = tk.Tk
    STYLE = {}

# Токенизатор предложений: razdel или fall back на NLTK
try:
    from razdel import sentenize
    tokenize = lambda txt: [s.text for s in sentenize(txt)]
except ImportError:
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
    tokenize = sent_tokenize

# Обёртки для предсказания
try:
    from main import run_predict_transformer, run_predict_rnn
except ImportError:
    run_predict_transformer = run_predict_rnn = None

# Класс ленивой загрузки моделей
class SentenceClassifier:
    def __init__(self, backend: str = 'transformer'):
        self.backend = backend
        self._ready = False
    def _load(self):
        if self._ready:
            return
        if self.backend == 'transformer':
            if run_predict_transformer is None:
                raise RuntimeError('run_predict_transformer не найден')
            self.predict_fn = run_predict_transformer
        else:
            if run_predict_rnn is None:
                raise RuntimeError('run_predict_rnn не найден')
            self.predict_fn = run_predict_rnn
        self._ready = True
    def predict(self, sentences: list[str]) -> list[int]:
        self._load()
        return [self.predict_fn(s) for s in sentences]

# Основное приложение GUI
class HighlighterApp(BaseApp):
    def __init__(self, **kwargs):
        super().__init__(**STYLE)
        self.title('Sarcasm / Joke Highlighter')
        self.geometry('1000x700')
        self._build_ui()
        nltk.download('punkt', quiet=True)

    def _build_ui(self):
        # Верхняя панель
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ttk.Label(top, text='Backend:').pack(side=tk.LEFT)
        self.backend_var = tk.StringVar(value='transformer')
        ttk.Combobox(
            top, textvariable=self.backend_var,
            values=['transformer', 'rnn'], state='readonly', width=14
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text='Открыть файл', command=self.on_open).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text='Найти юмор', command=self.on_start).pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(top, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Текстовое поле
        self.text = tk.Text(self, wrap='word')
        self.text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.text.bind('<Control-KeyPress>', self._on_ctrl_key)

        # Теги подсветки
        self.text.tag_configure('humor', background='#C8E6C9')



    def _show_context_menu(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def on_open(self):
        path = filedialog.askopenfilename(
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
        )
        if not path:
            return
        content = Path(path).read_text(encoding='utf-8')
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, content)

    def on_start(self):
        txt = self.text.get('1.0', tk.END).strip()
        if not txt:
            messagebox.showwarning('Пустой текст', 'Пожалуйста, вставьте или откройте текст.')
            return
        self.text.tag_remove('humor', '1.0', tk.END)
        backend = self.backend_var.get()
        self.progress.start(8)
        threading.Thread(target=self._process, args=(txt, backend), daemon=True).start()

    def _process(self, txt, backend):
        sentences = tokenize(txt)
        offsets = self._compute_offsets(txt, sentences)
        try:
            preds = SentenceClassifier(backend).predict(sentences)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Ошибка', str(e)))
            self.after(0, self.progress.stop)
            return
        self.after(0, self._apply_highlight, offsets, preds)
        self.after(0, self.progress.stop)

    def _compute_offsets(self, text, sentences):
        offs, start = [], 0
        for s in sentences:
            idx = text.find(s, start)
            if idx < 0:
                idx = start
            offs.append((idx, idx + len(s)))
            start = idx + len(s)
        return offs

    def _apply_highlight(self, offsets, preds):
        for (s, e), p in zip(offsets, preds):
            if p == 1:
                tag = 'humor'
            else:
                continue
            start = f'1.0+{s}c'
            end   = f'1.0+{e}c'
            self.text.tag_add(tag, start, end)

    def _on_ctrl_key(self, event):
        CTRL_MASK = 0x4
        V_SCANCODE = 86
        if event.state & CTRL_MASK and event.keycode == V_SCANCODE:
            self.text.event_generate('<<Paste>>')
            return 'break'

if __name__ == '__main__':
    app = HighlighterApp()
    app.mainloop()

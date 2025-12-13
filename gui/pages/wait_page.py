# gui/pages/wait_page.py
import ttkbootstrap as ttk
import tkinter as tk

class WaitPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.progress = ttk.Progressbar(self, mode="indeterminate", bootstyle="info-striped")
        self.progress.pack(fill="x", padx=20, pady=10)
        self.progress.start(10)

        self.label = ttk.Label(self, text="Initializing Genetic Algorithm...", font=("Helvetica", 12))
        self.label.pack(pady=10)

        self.status_text = tk.StringVar(value="Waiting for updates...")
        ttk.Label(self, textvariable=self.status_text).pack(pady=5)

    def update(self, event_type, data):
        """Left intentionally blank: observer pattern removed.

        The GUI no longer expects the GA service to push progress updates
        via an observer. Completion is handled by the controller when the
        GA calls the provided completion callback. If needed, use
        `set_status(text)` to change the status string from the UI thread.
        """

    def set_status(self, text: str):
        """Set the status label text (call from UI thread)."""
        self.status_text.set(text)
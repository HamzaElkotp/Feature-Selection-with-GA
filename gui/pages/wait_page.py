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
        """Called automatically by the GA via observer pattern."""
        if event_type == "progress":
            gen = data["generation"]
            fit = data["fitness"]
            self.status_text.set(f"Generation {gen}: current best fitness = {fit:.2f}")
        elif event_type == "complete":
            # Optionally stop animation when complete
            self.progress.stop()
            self.status_text.set("GA complete. Preparing results...")

            # At this point, we rely on controller to trigger results page
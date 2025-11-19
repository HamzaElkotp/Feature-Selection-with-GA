# gui/pages/start_page.py
import ttkbootstrap as ttk


class StartPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        # an entry to choose the dataset from the local storge

        ttk.Label(
            self,
            text="Choose the Mode",
            font=("Helvetica", 16, "bold")
        ).pack(pady=20)

        ttk.Button(
            self,
            text="Deterministic Mode",
            bootstyle="success",
            command=self._on_deterministic
        ).pack(pady=10)

        ttk.Button(
            self,
            text="Non-Deterministic Mode",
            bootstyle="info",
            command=self._on_nondeterministic
        ).pack(pady=10)

    def _on_deterministic(self):
        """Go to deterministic InputPage."""
        self.master.show_input_page(mode="deterministic")

    def _on_nondeterministic(self):
        """Go to NonDeterministic page."""
        self.master.show_nondeterministic_page()
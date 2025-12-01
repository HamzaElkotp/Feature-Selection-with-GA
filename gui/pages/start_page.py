import ttkbootstrap as ttk
from tkinter import filedialog
from pathlib import Path
from gui.utils.validator import validate_dataset_path_inline

class StartPage(ttk.Frame):

    def __init__(self, master):
        super().__init__(master)

        ttk.Label(self, text="Select Dataset", font=("Helvetica", 14, "bold")).pack(pady=(40, 10))

        # ----------------------------------------------------------------------
        # Dataset path entry + inline error label
        self.dataset_var = ttk.StringVar(value="/home/ahmed/Downloads/20230044-Sheet1.pdf")
        self.dataset_entry = ttk.Entry(self, textvariable=self.dataset_var, width=60)
        self.dataset_entry.pack(side="top", pady=5)

        self.error_label = ttk.Label(
            self, text="", bootstyle="danger", font=("", 9, "italic")
        )
        self.error_label.pack(pady=(0, 10))
        # ----------------------------------------------------------------------

        ttk.Button(self, text="Browse Dataset", bootstyle="info",
                   command=self._select_file).pack(pady=(0, 20))
        ttk.Button(self, text="Deterministic Mode", bootstyle="success",
                   command=self._on_deterministic).pack(pady=5)
        ttk.Button(self, text="Non‑Deterministic Mode", bootstyle="secondary",
                   command=self._on_nondeterministic).pack(pady=5)

        # validate on focus-out (manual typing)
        validate_cmd = (self.register(self._validate_inline), "%P")
        self.dataset_entry.configure(validate="focusout", validatecommand=validate_cmd)

    # --------------------------------------------------------------------------
    def _validate_inline(self, proposed_value: str):
        """Trigger validation when user leaves entry."""
        is_valid, msg = validate_dataset_path_inline(proposed_value)
        if not is_valid:
            self.dataset_entry.configure(bootstyle="danger")
            self.error_label.config(text=msg)
        else:
            self.dataset_entry.configure(bootstyle="default")
            self.error_label.config(text="")
        return is_valid

    # --------------------------------------------------------------------------
    def _select_file(self):
        
        path = filedialog.askopenfilename(
            title="Select Dataset",
            initialdir=Path.home(),
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

        if path:
            # validator from shared/validator.py
            is_valid, msg = validate_dataset_path_inline(path)
            if not is_valid:
                self.dataset_entry.configure(bootstyle="danger")
                self.error_label.config(text=msg)
                return

            self.dataset_entry.configure(bootstyle="default")
            self.error_label.config(text="")
            self.dataset_var.set(path)
            self.master.app_context.dataset_path = Path(path)

    # --------------------------------------------------------------------------
    def _on_deterministic(self):
        path_str = self.dataset_var.get().strip()
        is_valid, msg = validate_dataset_path_inline(path_str)
        if not is_valid:
            self.dataset_entry.configure(bootstyle="danger")
            self.error_label.config(text=msg)
            return

        self.dataset_entry.configure(bootstyle="default")
        self.error_label.config(text="")
        self.master.app_context.dataset_path = Path(path_str)
        self.master.app_context.mode = "deterministic"
        self.master.show_input_page(mode="deterministic")

    # --------------------------------------------------------------------------
    def _on_nondeterministic(self):
        path_str = self.dataset_var.get().strip()
        is_valid, msg = validate_dataset_path_inline(path_str)
        if not is_valid:
            self.dataset_entry.configure(bootstyle="danger")
            self.error_label.config(text=msg)
            return

        self.dataset_entry.configure(bootstyle="default")
        self.error_label.config(text="")
        self.master.app_context.dataset_path = Path(path_str)
        self.master.app_context.mode = "non-deterministic"
        self.master.show_nondeterministic_page()
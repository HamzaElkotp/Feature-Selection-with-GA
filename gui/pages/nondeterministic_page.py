import ttkbootstrap as ttk


class NonDeterministicPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        ttk.Label(
            self,
            text="Non-Deterministic Setup",
            font=("Helvetica", 14, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=20)

        # Two dropdown lists
        ttk.Label(self, text="Termination Condition:").grid(row=1, column=0, sticky="e")
        self.termination_var = ttk.StringVar(value="Max Generations")
        ttk.Combobox(self, textvariable=self.termination_var,
                     values=["Max Generations", "Score Threshold", "Convergence"]).grid(row=1, column=1, pady=5, sticky="w")

        ttk.Button(
            self,
            text="Continue",
            command=self._continue
        ).grid(row=3, column=0, columnspan=2, pady=15)

    def _continue(self):
        # Pass info to controller or InputPage
        self.master.show_input_page(mode="nondeterministic", extra_data={
            "termination_condition": self.termination_var.get(),
            "dataset": self.dataset_var.get()
        })
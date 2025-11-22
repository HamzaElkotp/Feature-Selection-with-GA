import json, tkinter as tk, ttkbootstrap as ttk
from gui.controller import InputController
from gui.components.widgets import numeric_spinbox

class InputPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.mode_var = tk.StringVar(value="Deterministic Input Mode")
        self.elitism_var = tk.DoubleVar(value=5)
        self.mutation_var = tk.DoubleVar(value=5)
        self.alpha_var = tk.DoubleVar(value=1)
        self.beta_var = tk.DoubleVar(value=1)

        # more values to be added (check "models.py")
        numeric_spinbox(self, self.elitism_var, 0, 25, step=0.05, label_text="Elitism %:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        numeric_spinbox(self, self.mutation_var, 0, 25, step=0.05, label_text="Mutation %:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        numeric_spinbox(self, self.alpha_var, 0, 2, step=0.05, label_text="Alpha:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        numeric_spinbox(self, self.beta_var, 0, 2, step=0.05, label_text="Beta:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(self, text="Termination Condition:").grid(row=4, column=0, sticky="w")
        self.termination_var = ttk.StringVar(value="Max Generations")
        ttk.Combobox(self, textvariable=self.termination_var,
                     values=["Max Generations", "Score Threshold", "Convergence"]).grid(row=4, column=1,padx=10, pady=5, sticky="w")
        ttk.Button(self, text="Run", command=self.on_run).grid(row=5, column=0,padx=10, pady=5, sticky="w")

    def on_run(self):
        
        ctx = self.master.app_context
        dataset_path = str(ctx.dataset_path)
        mode = ctx.mode

        data = {
            "mode":mode,
            "dataset_path": dataset_path,
            "termination_condition": str(self.termination_var),
            "select_method": "Roulette",
            "cross_method": "Uniform",
            "mutation_method": "Swap",
            "elitism_percent": self.elitism_var.get(),
            "mutation_percent": self.mutation_var.get(),
            "alpha": self.alpha_var.get(),
            "beta": self.beta_var.get()
        }
        json_data = json.dumps(data)
        controller = InputController(gui_context=self.master)
        controller.handle_user_input(json_data)
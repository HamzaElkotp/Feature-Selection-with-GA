import json
import ttkbootstrap as ttk

from gui.controller import InputController


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
        '''self.master.show_input_page(mode="nondeterministic", extra_data={
            "termination_condition": self.termination_var.get(),
            "dataset_path": self.master.app_context.dataset_path
        })'''

        ctx = self.master.app_context
        dataset_path = str(ctx.dataset_path)
        mode = ctx.mode

        data = {
            "mode":mode,
            "termination_condition": str(self.termination_var),
            "dataset_path": dataset_path,
            #"select_method": "Roulette",
            #"cross_method": "Uniform",
            #"mutation_method": "Swap",
            #"elitism_percent": self.elitism_var.get(),
            #"mutation_percent": self.mutation_var.get(),
            #"alpha": self.alpha_var.get(),
            #"beta": self.beta_var.get()
        }
        json_data = json.dumps(data)
        controller = InputController(gui_context=self.master)
        controller.handle_user_input(json_data)
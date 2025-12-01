import json, tkinter as tk, ttkbootstrap as ttk
from gui.controller import InputController
from gui.components.widgets import numeric_spinbox
from gui.components.widgets import dropdown_combobox

from interfaces.enums import RunMode, SelectionMethod, CrossoverMethod, MutationMethod, TerminationCondition
from interfaces.types import GAParameters, RunGAParameters

class InputPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.mode_var: str = tk.StringVar(value="Deterministic Input Mode")
        self.elitism_var: float = tk.DoubleVar(value=5)
        self.mutation_var: float = tk.DoubleVar(value=5)
        self.alpha_var: float = tk.DoubleVar(value=1)
        self.beta_var: float = tk.DoubleVar(value=1)
        self.select_method_var: str = tk.StringVar()
        self.cross_method_var: str = tk.StringVar()
        self.mutation_method_var: str = tk.StringVar()
        self.termination_condition: str = tk.StringVar()


        # more values to be added (check "models.py")
        numeric_spinbox(self, self.elitism_var, 0, 25, step=0.05, label_text="Elitism %:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        numeric_spinbox(self, self.mutation_var, 0, 25, step=0.05, label_text="Mutation %:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        numeric_spinbox(self, self.alpha_var, 0, 2, step=0.05, label_text="Alpha:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        numeric_spinbox(self, self.beta_var, 0, 2, step=0.05, label_text="Beta:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        dropdown_combobox(
            self,
            self.select_method_var,
            ["Roulette", "Tournament", "Rank"],
            label_text="Selection Method:"
        ).grid(row=4, column=0, padx=10, pady=5, sticky="w")

        dropdown_combobox(
            self,
            self.cross_method_var,
            ["Uniform", "Single Point", "Two Point"],
            label_text="Crossover Method:"
        ).grid(row=5, column=0, padx=10, pady=5, sticky="w")

        dropdown_combobox(
            self,
            self.mutation_method_var,
            ["Swap", "Scramble", "Inversion"],
            label_text="Mutation Method:"
        ).grid(row=6, column=0, padx=10, pady=5, sticky="w")

        dropdown_combobox(
            self,
            self.termination_condition,
            ["Max Generations", "Score Threshold", "Convergence"],
            label_text="Termination Condition:"
        ).grid(row=7, column=0, padx=10, pady=5, sticky="w")

        ttk.Button(self, text="Run", command=self.on_run).grid(row=8, column=0,padx=10, pady=5, sticky="w")
        
        ttk.Button(self, text="Back to Start", command=self.master.show_start_page).grid(row=9, column=0,padx=10, pady=5, sticky="w")

    def on_run(self):
        context = self.master.app_context
        dataset_path = str(context.dataset_path) if context.dataset_path else ""

        # map UI labels to Enums
        select_map = {
            "Roulette": SelectionMethod.ROULETTE,
            "Rank": SelectionMethod.RANKING,
            "Tournament": SelectionMethod.TOURNAMENT,
        }
        cross_map = {
            "Uniform": CrossoverMethod.MULTI_POINT,
            "Single Point": CrossoverMethod.SINGLE_POINT,
            "Two Point": CrossoverMethod.MULTI_POINT,
        }
        mutation_map = {
            "Swap": MutationMethod.BIT_FLIP,
            "Scramble": MutationMethod.INVERSION,
            "Inversion": MutationMethod.INVERSION,
        }
        term_map = {
            "Max Generations": TerminationCondition.AFTER_N_GENERATIONS,
            "Score Threshold": TerminationCondition.AFTER_FITNESS_REACHES_N,
            "Convergence": TerminationCondition.NO_IMPROVEMENT_SINCE_N_GENERATIONS,
        }

        sel_enum = select_map.get(self.select_method_var.get(), SelectionMethod.ROULETTE)
        cross_enum = cross_map.get(self.cross_method_var.get(), CrossoverMethod.MULTI_POINT)
        mut_enum = mutation_map.get(self.mutation_method_var.get(), MutationMethod.BIT_FLIP)
        term_enum = term_map.get(self.termination_condition.get(), TerminationCondition.AFTER_N_GENERATIONS)

        ga_params = GAParameters(
            selection_method=sel_enum,
            crossover_method=cross_enum,
            mutation_method=mut_enum,
            termination_condition=term_enum,
            termination_condition_n=0.0,
            elitism_percent=self.elitism_var.get(),
            mutation_percent=self.mutation_var.get(),
            alpha=self.alpha_var.get(),
            beta=self.beta_var.get(),
        )

        run_params = RunGAParameters(
            dataset_file_path=dataset_path,
            mode=RunMode.DIM,
            ga_parameters=ga_params,
        )

        # store typed params in app context for later inspection
        context.ga_parameters = ga_params

        controller = InputController(gui_context=self.master)
        controller.handle_user_input(run_params)
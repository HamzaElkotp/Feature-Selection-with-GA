import json, tkinter as tk, ttkbootstrap as ttk
from gui.controller import InputController
from gui.components.widgets import numeric_spinbox, dropdown_combobox, text_entry

from interfaces.enums import RunMode, SelectionMethod, MutationMethod #, CrossoverMethod, TerminationCondition
from interfaces.types import GAParameters, RunGAParameters

class InputPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.mode_var: tk.StringVar = tk.StringVar(value="Deterministic Input Mode")

        #Genetic Algorithm Parameters
        self.elitism_var: tk.IntVar = tk.IntVar(value=1)
        self.mutation_var: tk.IntVar = tk.IntVar(value=1)
        self.cross_k_points: tk.IntVar = tk.IntVar(value=2)
        self.initial_population_size: tk.IntVar = tk.IntVar(value=20)
        #Fitness / Model Parameters
        self.alpha_var: tk.DoubleVar = tk.DoubleVar(value=1)
        self.beta_var: tk.DoubleVar = tk.DoubleVar(value=1)
        self.num_of_generations: tk.IntVar = tk.IntVar(value=30)
        #Dataset Configuration
        self.result_column_name: tk.StringVar = tk.StringVar()
        #Operators
        self.select_method_var: tk.StringVar = tk.StringVar()
        self.mutation_method_var: tk.StringVar = tk.StringVar()


        ttk.Label(
            self,
            text="Genetic Algorithm Parameters",
            bootstyle="info"
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        (numeric_spinbox(
            self,
            self.elitism_var,
            0,
            100,
            step=1,
            label_text="Elitism %: "
        ).grid(row=1, column=0, padx=10, pady=5, sticky="w"))

        numeric_spinbox(
            self,
            self.mutation_var,
            0,
            100,
            step=1,
            label_text="Mutation %: "
        ).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        numeric_spinbox(
            self,
            self.cross_k_points,
            1,
            20,
            step=1,
            label_text="K Cross-Overs Points: "
        ).grid(row=2, column=0, padx=10, pady=5, sticky="w")

        numeric_spinbox(
            self,
            self.initial_population_size,
            4,
            5000,
            step=1,
            label_text="Initial Population Size: "
        ).grid(row=2, column=1, padx=10, pady=5, sticky="w")


        """""""""""""""""""""""""""
        """""""""""""""""""""""""""

        ttk.Label(
            self,
            text="Fitness / Model Parameters",
            bootstyle="info"
        ).grid(row=4, column=0, padx=10, pady=5, sticky="w")

        numeric_spinbox(
            self,
            self.alpha_var,
            0,
            2,
            step=0.05,
            label_text="Alpha: "
        ).grid(row=5, column=0, padx=10, pady=5, sticky="w")

        numeric_spinbox(
            self,
            self.beta_var,
            0,
            2,
            step=0.05,
            label_text="Beta: "
        ).grid(row=5, column=1, padx=10, pady=5, sticky="w")

        numeric_spinbox(
            self,
            self.num_of_generations,
            1,
            5000,
            step=1,
            label_text="Number Of Generations: "
        ).grid(row=6, column=0, padx=10, pady=5, sticky="w")

        """""""""""""""""""""""""""
        """""""""""""""""""""""""""

        ttk.Label(
            self,
            text="GA Operators",
            bootstyle="info"
        ).grid(row=8, column=0, padx=10, pady=5, sticky="w")

        dropdown_combobox(
            self,
            self.select_method_var,
            ["Roulette", "Tournament", "Random"],
            label_text="Selection Method:"
        ).grid(row=9, column=0, padx=10, pady=5, sticky="w")

        dropdown_combobox(
            self,
            self.mutation_method_var,
            ["Bit Flip", "Bit String Complement", "Bit String Reverse", "Bit String Rotation"],
            label_text="Mutation Method:"
        ).grid(row=9, column=1, padx=10, pady=5, sticky="w")

        """""""""""""""""""""""""""
        """""""""""""""""""""""""""

        ttk.Label(
            self,
            text="Dataset Configuration",
            bootstyle="info"
        ).grid(row=12, column=0, padx=10, pady=5, sticky="w")

        text_entry(
            self,
            self.result_column_name,
            label_text="Dataset Result (Target) Column Name",
            placeholder="ex: Diagnose",
            width=15
        ).grid(row=13, column=0, padx=10, pady=5, sticky="w")



        ttk.Button(
            self,
            text="Back to Start",
            command=self.master.show_start_page
        ).grid(row=14, column=0, padx=10, pady=5, sticky="w")

        ttk.Button(
            self,
            text="Run",
            command=self.on_run
        ).grid(row=14, column=2,padx=10, pady=5, sticky="w")


    def on_run(self):
        context = self.master.app_context
        dataset_path = str(context.dataset_path) if context.dataset_path else ""

        # map UI labels to Enums
        select_map = {
            "Roulette": SelectionMethod.ROULETTE,
            "Tournament": SelectionMethod.TOURNAMENT,
            "Random": SelectionMethod.RANDOM,
        }
        # cross_map = {
        #     "Uniform": CrossoverMethod.MULTI_POINT,
        #     "Single Point": CrossoverMethod.SINGLE_POINT,
        #     "Two Point": CrossoverMethod.MULTI_POINT,
        # }
        mutation_map = {
            "Bit Flip": MutationMethod.Bit_Flip,
            "Bit String Complement": MutationMethod.Bit_String_Complement,
            "Bit String Reverse": MutationMethod.Bit_String_Reverse,
            "Bit String Rotation": MutationMethod.Bit_String_Rotation,
        }
        # term_map = {
        #     "Max Generations": TerminationCondition.AFTER_N_GENERATIONS,
        #     "Score Threshold": TerminationCondition.AFTER_FITNESS_REACHES_N,
        #     "Convergence": TerminationCondition.NO_IMPROVEMENT_SINCE_N_GENERATIONS,
        # }

        sel_enum = select_map.get(self.select_method_var.get(), SelectionMethod.ROULETTE)
        # cross_enum = cross_map.get(self.cross_k_points.get(), CrossoverMethod.MULTI_POINT)
        mut_enum = mutation_map.get(self.mutation_method_var.get(), MutationMethod.Bit_Flip)
        # term_enum = term_map.get(self.termination_condition.get(), TerminationCondition.AFTER_N_GENERATIONS)

        ga_params = GAParameters(
            elitism_percent = self.elitism_var.get(),
            mutation_percent = self.mutation_var.get(),
            crossover_k_points = self.cross_k_points.get(),
            initial_population_size = self.initial_population_size.get(),

            alpha = self.alpha_var.get(),
            beta = self.beta_var.get(),
            num_of_generations = self.num_of_generations.get(),

            result_col_name = self.result_column_name.get(),

            selection_method=sel_enum,
            mutation_method=mut_enum,
            # crossover_method=cross_enum,
            # termination_condition=term_enum,
            # termination_condition_n=0.0,
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
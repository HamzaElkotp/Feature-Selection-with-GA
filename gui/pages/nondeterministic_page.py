import json
import ttkbootstrap as ttk
from ttkbootstrap.tableview import Tableview

from gui.controller import InputController


class NonDeterministicPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # Page title
        ttk.Label(self, text="Non-Deterministic Setup", font=("Helvetica", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(12, 8)
        )

        # card container to give padding and a distinct area for controls
        card = ttk.Frame(self, padding=12)
        card.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 8))

        # table of constant values for non-deterministic mode
        # provide column width & anchor metadata for a clearer layout
        columns = (
            {"text": "Alpha", "width": 80, "anchor": "center"},
            {"text": "Beta", "width": 80, "anchor": "center"},
            {"text": "Mutation Rate", "width": 140, "anchor": "center"},
            {"text": "Crossover Rate", "width": 140, "anchor": "center"},
            {"text": "Population Size", "width": 120, "anchor": "center"},
        )

        # sample rows: expand to 5 rows (original had one) to make the table more practical
        rowdata = [
            ("0.5", "2", "0.12", "0.12", "50"),
            ("0.3", "1.5", "0.10", "0.15", "60"),
            ("0.7", "2.0", "0.08", "0.20", "40"),
            ("0.4", "1.8", "0.05", "0.10", "80"),
            ("0.6", "2.2", "0.15", "0.18", "30"),
        ]

        self.constants_table = Tableview(
            card,
            coldata=columns,
            rowdata=rowdata,
            height=8,
            bootstyle="info",
            paginated=True,
        )
        # make the table fill the card
        self.constants_table.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=6, pady=(6, 12))

        # allow the card to expand the table when the window resizes
        card.grid_rowconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)

        # Termination controls: placed below the table, aligned to the left
        ttk.Label(card, text="Termination Condition:").grid(row=1, column=0, sticky="w", padx=(2, 6))
        self.termination_var = ttk.StringVar(value="Max Generations")
        ttk.Combobox(card, textvariable=self.termination_var,
                     values=["Max Generations", "Score Threshold", "Convergence"], width=18).grid(row=1, column=1, pady=6, sticky="e")

        # Continue button centered below the card
        ttk.Button(self, text="Continue", command=self._continue).grid(row=2, column=0, columnspan=2, pady=(6, 12))

        # Back button at bottom-left
        ttk.Button(self, text="Back to Start", command=self.master.show_start_page).grid(row=3, column=0, padx=10, pady=6, sticky="w")

        # allow the outer page to expand
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _continue(self):
        # Pass info to controller
        
        ctx = self.master.app_context
        dataset_path = ctx.dataset_path
        mode = ctx.mode
        # Build typed RunGAParameters (reuse the same GAParameters dataclass)
        from interfaces.enums import TerminationCondition, RunMode, SelectionMethod, CrossoverMethod, MutationMethod
        from interfaces.types import GAParameters, RunGAParameters

        # read the first row of the constants table (alpha/beta etc) if present
        alpha = 1.0
        beta = 1.0
        elitism = 5.0
        mutation_percent = 5.0
        try:
            children = self.constants_table.view.get_children()
            if children:
                vals = self.constants_table.view.item(children[0])["values"]
                # expected order: Alpha, Beta, Mutation Rate, Crossover Rate, Population Size
                alpha = float(vals[0])
                beta = float(vals[1])
                mutation_percent = float(vals[2]) * 100 if float(vals[2]) <= 1 else float(vals[2])
                # keep elitism default
        except Exception:
            pass

        term_map = {
            "Max Generations": TerminationCondition.AFTER_N_GENERATIONS,
            "Score Threshold": TerminationCondition.AFTER_FITNESS_REACHES_N,
            "Convergence": TerminationCondition.NO_IMPROVEMENT_SINCE_N_GENERATIONS,
        }

        term_enum = term_map.get(self.termination_var.get(), TerminationCondition.AFTER_N_GENERATIONS)

        ga_params = GAParameters(
            selection_method=SelectionMethod.ROULETTE,
            crossover_method=CrossoverMethod.SINGLE_POINT,
            mutation_method=MutationMethod.BIT_FLIP,
            termination_condition=term_enum,
            termination_condition_n=0.0,
            elitism_percent=elitism,
            mutation_percent=mutation_percent,
            alpha=alpha,
            beta=beta,
        )

        run_params = RunGAParameters(dataset_file_path=str(dataset_path) if dataset_path else "",
                                     mode=RunMode.EBM,
                                     ga_parameters=ga_params)

        # store typed params in app context and hand to controller
        ctx.ga_parameters = ga_params
        controller = InputController(gui_context=self.master)
        controller.handle_user_input(run_params)
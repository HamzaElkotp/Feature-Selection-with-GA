import json, threading
from tkinter import messagebox
from core.models import DeterministicInputConfig
from core.validators import validate_config

class InputController:
    def __init__(self, gui_context):
        self.gui_context = gui_context
        self.ga = None

    def handle_user_input(self, json_input, observers=None):
        try:
            data = json.loads(json_input)
            config = DeterministicInputConfig(**data)
            validate_config(config)
        except Exception as e:
            messagebox.showerror("Config Error", str(e))
            return

        #self.ga = GeneticAlgorithm(config)
        if observers:
            for o in observers:
                self.ga.attach(o)

        if hasattr(self.gui_context, "show_wait_page"):
            self.gui_context.show_wait_page()

        threading.Thread(target=self.run_ga, daemon=True).start()
        

    def run_ga(self):
        results = self.ga.run()
        self.gui_context.after(0, self.gui_context.show_results_page, results)
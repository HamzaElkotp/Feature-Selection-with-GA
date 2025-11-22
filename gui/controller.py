# core/controller.py
import json
import threading
from tkinter import messagebox
from gui.models import InputConfig
from gui.utils.validator import validate_config
from gui.ga_runner import GeneticAlgorithm


class InputController:
    def __init__(self, gui_context):
        self.gui_context = gui_context
        self.ga = None

    def handle_user_input(self, json_input):
        try:
            data = json.loads(json_input)
            config = InputConfig(**data)
            validate_config(config)
        except Exception as e:
            messagebox.showerror("Config Error", str(e))
            return

        # Instantiate GA and attach observers
        self.ga = GeneticAlgorithm(config)

        # Show the Wait Page FIRST
        self.gui_context.show_wait_page()

        # The Wait Page itself becomes an observer
        wait_page = self.gui_context._current_page
        self.ga.attach(wait_page)

        # Run GA in background thread
        thread = threading.Thread(target=self.run_ga, daemon=True)
        thread.start()

    def run_ga(self):
        results = self.ga.run()

        # After final completion, trigger Results Page
        self.gui_context.after(0, self.gui_context.show_results_page, results)
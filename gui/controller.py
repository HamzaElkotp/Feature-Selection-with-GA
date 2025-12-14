import threading
from tkinter import messagebox

from interfaces.types import RunGAParameters, RunGAResult
from interfaces.enums import RunMode

from core.GA_functions import (
# Types
Merged_GA
)

class InputController:
    """Controller that accepts typed RunGAParameters and invokes the GA service.

    This no-longer expects JSON: callers (GUI pages) should construct a
    `RunGAParameters` instance and pass it here.
    """

    def __init__(self, gui_context):
        self.gui_context = gui_context

    def handle_user_input(self, run_params: RunGAParameters):
        # Show the Wait Page
        self.gui_context.show_wait_page()

        # Run GA in a background thread so the UI remains responsive
        thread = threading.Thread(target=self.run_ga, args=(run_params,), daemon=True)
        thread.start()

    def run_ga(self, run_params: RunGAParameters):
        # Use the ga_service attached to the application (implements GAInterface)
        # Define a completion callback the GA service will call when finished.
        def on_complete(results: Merged_GA):
            # Ensure we schedule UI updates on the main/UI thread
            self.gui_context.after(0, self.gui_context.show_results_page, results)

        try:
            # Pass the on_complete callback to the GA service. The service
            # implementation is expected to call this callback when it finishes.
            self.gui_context.ga_service.run_ga(run_params, on_complete=on_complete)
        except Exception as e:
            # surface error on UI thread
            self.gui_context.after(0, messagebox.showerror, "GA Error", str(e))
            return
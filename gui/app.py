import ttkbootstrap as ttk

from interfaces.api_contract import GAInterface
from shared.context import AppContext
from gui.pages.start_page import StartPage
from gui.pages.input_page import InputPage
from gui.pages.nondeterministic_page import NonDeterministicPage
from gui.pages.wait_page import WaitPage
from gui.pages.results_page import ResultsPage

from core.GA_functions import (
# Types
Merged_GA
)

class Application(ttk.Window):
    def __init__(self, ga_service: GAInterface):
        super().__init__(themename="darkly")
        self.title("Feature Selection for DTs Using GA")
        self.geometry("800x500")

        # The passed ga_service will work fine as long as it implements the GAInterface
        self.ga_service = ga_service

        # ----- Shared dataclass instance -----
        self.app_context = AppContext()
        self._current_page = None

        self.show_start_page()
        self.mainloop()

    def _switch_page(self, page_class, **kwargs):
        """Destroy current frame and show a new one."""
        if self._current_page:
            self._current_page.destroy()
        self._current_page = page_class(self, **kwargs)
        self._current_page.pack(fill="both", expand=True)

    # Navigation helpers
    def show_start_page(self):
        self._switch_page(StartPage)

    def show_input_page(self, mode="deterministic"):
        self.app_context.mode = mode
        #print("App Context:", self.app_context.summary())  # optional debug
        self._switch_page(InputPage)

    def show_nondeterministic_page(self):
        self.app_context.mode = "non-deterministic"
        self._switch_page(NonDeterministicPage)

    def show_wait_page(self):
        """Show waiting screen while GA runs."""
        self._switch_page(WaitPage)

    def show_results_page(self, dt_result:Merged_GA, rf_result:Merged_GA):
        """Show GA results page."""
        self._switch_page(ResultsPage, dt_result=dt_result, rf_result=rf_result)       

# if __name__ == "__main__":
#     Application()
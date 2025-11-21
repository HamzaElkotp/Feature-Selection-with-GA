import ttkbootstrap as ttk
from shared.context import AppContext
from gui.pages.start_page import StartPage
from gui.pages.input_page import InputPage
from gui.pages.nondeterministic_page import NonDeterministicPage
from gui.pages.wait_page import WaitPage
from gui.pages.results_page import ResultsPage


class Application(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Feature Selection with GA")
        self.geometry("800x500")

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

    def show_results_page(self, results):
        """Show GA results page."""
        self._switch_page(ResultsPage, results=results)       

if __name__ == "__main__":
    Application()
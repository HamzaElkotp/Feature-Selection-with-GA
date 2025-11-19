import ttkbootstrap as ttk
from gui.pages.start_page import StartPage
from gui.pages.input_page import InputPage
from gui.pages.nondeterministic_page import NonDeterministicPage


class Application(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Feature Selection with GA")
        self.geometry("800x500")

        self._current_page = None
        self.show_start_page()

        self.mainloop()

    def _switch_page(self, new_page_class, **kwargs):
        """Destroy current frame and show a new one."""
        if self._current_page:
            self._current_page.destroy()
        self._current_page = new_page_class(self, **kwargs)
        self._current_page.pack(fill="both", expand=True)

    # Page navigation helpers
    def show_start_page(self):
        self._switch_page(StartPage)

    def show_input_page(self, mode="deterministic", extra_data=None):
        print(f"Starting GA in {mode} mode, extra data={extra_data}")
        self._switch_page(InputPage)

    def show_nondeterministic_page(self):
        self._switch_page(NonDeterministicPage)

        
if __name__ == "__main__":
    Application()
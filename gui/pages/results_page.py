import ttkbootstrap as ttk
import tkinter as tk

class ResultsPage(ttk.Frame):
    def __init__(self, master, results=None):
        super().__init__(master)
        ttk.Label(self, text="Results Page", font=("Helvetica", 16, "bold")).pack(pady=20)

        if results:
            result_text = tk.Text(self, height=15, width=70)
            result_text.pack(padx=20, pady=20)
            result_text.insert("end", str(results))
            result_text.configure(state="disabled")

        ttk.Button(self, text="Back to Start", command=master.show_start_page).pack(pady=10)
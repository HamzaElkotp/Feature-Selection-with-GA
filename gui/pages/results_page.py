import ttkbootstrap as ttk
import tkinter as tk
from ttkbootstrap.constants import *

from typing import List, Dict
from core.GA_functions import Chromosome, Merged_Generation, Merged_GA

# ------------------ Matplotlib setup ------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# =====================================================
# Scrollable container
# =====================================================
class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=VERTICAL, command=canvas.yview)

        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Mouse wheel support
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * int(e.delta / 120), "units"))

# =====================================================
# Results Page
# =====================================================
class ResultsPage(ttk.Frame):

    def __init__(self, master, results: Merged_GA):
        super().__init__(master)

        scroll = ScrollableFrame(self)
        scroll.pack(fill=BOTH, expand=True)

        self.container = scroll.scrollable_frame
        self.results = results

        ttk.Label(
            self.container,
            text="GA Results",
            font=("Helvetica", 18, "bold")
        ).pack(pady=(15, 20))

        if HAS_MATPLOTLIB:
            self._build_plots(self.container)
        else:
            ttk.Label(self.container, text="Matplotlib is not available").pack()

        self._build_tables(self.container)

    # =====================================================
    # -------------------- PLOTS --------------------------
    # =====================================================
    def _build_plots(self, parent: ttk.Frame):

        gens = list(range(len(self.results)))

        gen_sizes = [g["gen_size"] for g in self.results]
        best_fitness = [g["best_chromosome"]["fitness"] for g in self.results]
        worst_fitness = [g["worst_chromosome"]["fitness"] for g in self.results]
        avg_fitness = [g["average_generations_fitness"] for g in self.results]
        ones_count = [sum(g["best_chromosome"]["bit_string"]) for g in self.results]

        # ---- Row 1: Generation Size & Best/Worst Fitness ----
        row1 = ttk.Frame(parent)
        row1.pack(fill=BOTH, expand=True, pady=10)

        fig1 = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig1.add_subplot(111)
        ax1.plot(gens, gen_sizes, marker="o")
        ax1.set_title("Generation Size")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Size")
        ax1.grid(True, alpha=0.4)

        canvas1 = FigureCanvasTkAgg(fig1, master=row1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        fig2 = Figure(figsize=(6, 4), dpi=100)
        ax2 = fig2.add_subplot(111)
        ax2.plot(gens, best_fitness, label="Best Fitness")
        ax2.plot(gens, worst_fitness, label="Worst Fitness")
        ax2.set_title("Best & Worst Fitness")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Fitness")
        ax2.legend()
        ax2.grid(True, alpha=0.4)

        canvas2 = FigureCanvasTkAgg(fig2, master=row1)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        # ---- Row 2: Average Fitness & Selected Features ----
        row2 = ttk.Frame(parent)
        row2.pack(fill=BOTH, expand=True, pady=10)

        fig3 = Figure(figsize=(6, 4), dpi=100)
        ax3 = fig3.add_subplot(111)
        ax3.plot(gens, avg_fitness, marker="o", color="orange")
        ax3.set_title("Average Fitness Across Generations")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Average Fitness")
        ax3.grid(True, alpha=0.4)

        canvas3 = FigureCanvasTkAgg(fig3, master=row2)
        canvas3.draw()
        canvas3.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        fig4 = Figure(figsize=(6, 4), dpi=100)
        ax4 = fig4.add_subplot(111)
        ax4.plot(gens, ones_count, marker="o")
        stable = self._find_stable_point(ones_count)
        if stable is not None:
            ax4.axvline(stable, linestyle="--", alpha=0.7)
        ax4.set_title("Number of Selected Features (Best Chromosome)")
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Feature Count")
        ax4.grid(True, alpha=0.4)

        canvas4 = FigureCanvasTkAgg(fig4, master=row2)
        canvas4.draw()
        canvas4.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        # ---- Row 3: Best Fitness Stabilization (full width) ----
        row3 = ttk.Frame(parent)
        row3.pack(fill=BOTH, expand=True, pady=10)

        fig5 = Figure(figsize=(12, 4), dpi=100)
        ax5 = fig5.add_subplot(111)
        ax5.plot(gens, best_fitness, marker="o")
        stable_fitness = self._find_stable_point(best_fitness)
        if stable_fitness is not None:
            ax5.axvline(stable_fitness, linestyle="--", alpha=0.7)
        ax5.set_title("Best Fitness Stabilization")
        ax5.set_xlabel("Generation")
        ax5.set_ylabel("Fitness")
        ax5.grid(True, alpha=0.4)

        canvas5 = FigureCanvasTkAgg(fig5, master=row3)
        canvas5.draw()
        canvas5.get_tk_widget().pack(fill=BOTH, expand=True, padx=5)

    # =====================================================
    # -------------------- TABLES -------------------------
    # =====================================================
    def _build_tables(self, parent: ttk.Frame):

        tables_row = ttk.Frame(parent)
        tables_row.pack(fill=BOTH, expand=True, padx=10, pady=15)

        left = ttk.Frame(tables_row)
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        right = ttk.Frame(tables_row)
        right.pack(side=LEFT, fill=Y, padx=5)  # fit width

        # ---------- Table 1: Top 10 Best Chromosomes ----------
        ttk.Label(left, text="Top 10 Best Chromosomes", font=("Helvetica", 12, "bold")).pack(pady=8)
        cols = ("Generation", "Fitness", "Selected Features", "Decoded Value")
        table1 = ttk.Treeview(left, columns=cols, show="headings", height=18)
        for c in cols:
            table1.heading(c, text=c)
            table1.column(c, anchor=CENTER)
        table1.pack(fill=BOTH, expand=True)

        all_best = [(i, g["best_chromosome"]) for i, g in enumerate(self.results)]
        top10 = sorted(all_best, key=lambda x: x[1]["fitness"], reverse=True)[:10]

        for gen, chrom in top10:
            table1.insert(
                "",
                END,
                values=(
                    gen,
                    round(chrom["fitness"], 6),
                    sum(chrom["bit_string"]),
                    self._decode_chromosome(chrom["bit_string"]),
                ),
            )

        # ---------- Table 2: Feature Usage ----------
        ttk.Label(right, text="Feature Usage (Best Chromosomes)", font=("Helvetica", 12, "bold")).pack(pady=8)
        table2 = ttk.Treeview(right, columns=("Feature Index", "Usage Count"), show="headings", height=18)
        table2.heading("Feature Index", text="Feature Index")
        table2.heading("Usage Count", text="Usage Count")
        table2.column("Feature Index", anchor=CENTER)
        table2.column("Usage Count", anchor=CENTER)
        table2.pack(fill=Y)

        for idx, count in self._bit_index_statistics().items():
            table2.insert("", END, values=(idx, count))

    # =====================================================
    # -------------------- HELPERS ------------------------
    # =====================================================
    def _find_stable_point(self, values: List[float]):
        for i in range(1, len(values)):
            if all(values[j] == values[i] for j in range(i, len(values))):
                return i
        return None

    def _decode_chromosome(self, bit_string: List[int]) -> float:
        # Placeholder – replace with real decoder later
        return sum(bit_string)

    def _bit_index_statistics(self) -> Dict[int, int]:
        stats: Dict[int, int] = {}
        for g in self.results:
            for i, v in enumerate(g["best_chromosome"]["bit_string"]):
                if v == 1:
                    stats[i] = stats.get(i, 0) + 1
        return dict(sorted(stats.items()))

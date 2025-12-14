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
# Results Page
# =====================================================
class ResultsPage(ttk.Frame):

    def __init__(self, master, results: Merged_GA):
        super().__init__(master)
        self.results = results

        ttk.Label(self, text="GA Results", font=("Helvetica", 18, "bold")).pack(pady=10)

        # ---- Layout containers ----
        top = ttk.Frame(self)
        top.pack(fill=BOTH, expand=True)

        bottom = ttk.Frame(self)
        bottom.pack(fill=BOTH, expand=True, pady=10)

        if HAS_MATPLOTLIB:
            self._build_plots(top)
        else:
            ttk.Label(top, text="Matplotlib is not available").pack()

        self._build_tables(bottom)

    # =====================================================
    # -------------------- PLOTS --------------------------
    # =====================================================
    def _build_plots(self, parent: ttk.Frame):

        fig = Figure(figsize=(11, 8), dpi=100)

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        gens = list(range(len(self.results)))

        gen_sizes = [g["gen_size"] for g in self.results]
        best_fitness = [g["best_chromosome"]["fitness"] for g in self.results]
        worst_fitness = [g["worst_chromosome"]["fitness"] for g in self.results]
        avg_fitness = [g["average_generations_fitness"] for g in self.results]

        ones_count = [sum(g["best_chromosome"]["bit_string"]) for g in self.results]

        # ---- Plot 1: Generation size ----
        ax1.plot(gens, gen_sizes, marker="o")
        ax1.set_title("Generation Size")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Size")
        ax1.grid(True, alpha=0.4)

        # ---- Plot 2: Fitness comparison ----
        ax2.plot(gens, best_fitness, label="Best Fitness")
        ax2.plot(gens, worst_fitness, label="Worst Fitness")
        ax2.plot(gens, avg_fitness, label="Average Fitness")
        ax2.set_title("Fitness Across Generations")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Fitness")
        ax2.legend()
        ax2.grid(True, alpha=0.4)

        # ---- Plot 3: Number of 1s in best chromosome ----
        ax3.plot(gens, ones_count, marker="o")
        stable_ones = self._find_stable_point(ones_count)
        if stable_ones is not None:
            ax3.axvline(stable_ones, linestyle="--", alpha=0.7)

        ax3.set_title("Number of Selected Features in Best Chromosome")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Count of Features")
        ax3.grid(True, alpha=0.4)

        # ---- Plot 4: Best fitness stabilization ----
        ax4.plot(gens, best_fitness, marker="o")
        stable_fitness = self._find_stable_point(best_fitness)
        if stable_fitness is not None:
            ax4.axvline(stable_fitness, linestyle="--", alpha=0.7)

        ax4.set_title("Best Fitness Stabilization")
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Fitness")
        ax4.grid(True, alpha=0.4)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    # =====================================================
    # -------------------- TABLES -------------------------
    # =====================================================
    def _build_tables(self, parent: ttk.Frame):

        left = ttk.Frame(parent)
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        right = ttk.Frame(parent)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=5)

        # ---------- Table 1: Best 10 chromosomes ----------
        ttk.Label(left, text="Top 10 Best Chromosomes", font=("Helvetica", 12, "bold")).pack(pady=5)

        cols = ("Generation", "Fitness", "Number of Selected Features", "Selected Features")
        table1 = ttk.Treeview(left, columns=cols, show="headings", height=10)

        for c in cols:
            table1.heading(c, text=c)
            table1.column(c, anchor=CENTER)

        table1.pack(fill=BOTH, expand=True)

        all_best = [
            (i, g["best_chromosome"])
            for i, g in enumerate(self.results)
        ]

        top10 = sorted(all_best, key=lambda x: x[1]["fitness"], reverse=True)[:10]

        for gen_idx, chrom in top10:
            table1.insert(
                "",
                END,
                values=(
                    gen_idx,
                    round(chrom["fitness"], 5),
                    sum(chrom["bit_string"]),
                    self._decode_chromosome(chrom["bit_string"]),
                ),
            )

        # ---------- Table 2: Index contribution ----------
        ttk.Label(right, text="Feature Contribution (Best Chromosomes)", font=("Helvetica", 12, "bold")).pack(pady=5)

        table2 = ttk.Treeview(right, columns=("Feature", "Number of usage"), show="headings", height=10)
        table2.heading("Feature", text="Feature")
        table2.heading("Number of usage", text="Number of usage")

        table2.column("Number of usage", anchor=CENTER)
        table2.column("Number of usage", anchor=CENTER)

        table2.pack(fill=BOTH, expand=True)

        index_scores = self._bit_index_statistics()

        for idx, count in index_scores.items():
            table2.insert("", END, values=(idx, count))

    # =====================================================
    # -------------------- HELPERS ------------------------
    # =====================================================
    def _find_stable_point(self, values: List[float]):
        """Return index where values stop changing."""
        for i in range(1, len(values)):
            if all(values[j] == values[i] for j in range(i, len(values))):
                return i
        return None

    def _decode_chromosome(self, bit_string: List[int]) -> float:
        """
        Placeholder decoding logic.
        Replace later with real decoder.
        """
        return sum(bit_string)

    def _bit_index_statistics(self) -> Dict[int, int]:
        """Count how many times each bit index is 1 across all best chromosomes."""
        stats: Dict[int, int] = {}

        for g in self.results:
            bits = g["best_chromosome"]["bit_string"]
            for idx, val in enumerate(bits):
                if val == 1:
                    stats[idx] = stats.get(idx, 0) + 1

        return dict(sorted(stats.items()))

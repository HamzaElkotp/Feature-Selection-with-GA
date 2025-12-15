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
# Scrollable container (Tkinter best practice)
# =====================================================
class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)

        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Update canvas window width when canvas is resized
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Mouse wheel support
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 * int(e.delta / 120), "units"))

    def _on_canvas_configure(self, event):
        # Update the canvas window width to match the canvas width
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)


# =====================================================
# Results Page
# =====================================================
class ResultsPage(ttk.Frame):
    def __init__(self, master, dt_result: Merged_GA, rf_result: Merged_GA):
        super().__init__(master)

        print(dt_result)
        print(rf_result)

        self.dt_result = dt_result
        self.rf_result = rf_result

        # Create split view with PanedWindow
        paned = ttk.Panedwindow(self, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)

        # Left side - Single DT
        left_scroll = ScrollableFrame(paned)
        paned.add(left_scroll, weight=1)
        left_container = left_scroll.scrollable_frame

        # Right side - Random Forest
        right_scroll = ScrollableFrame(paned)
        paned.add(right_scroll, weight=1)
        right_container = right_scroll.scrollable_frame

        # Build left side (Single DT)
        self._build_side(left_container, "Single DT", self.dt_result)

        # Build right side (Random Forest)
        self._build_side(right_container, "Random Forest", self.rf_result)

    def _build_side(self, parent: ttk.Frame, title: str, result: Merged_GA):
        """Build plots and tables for one side of the split view"""
        # Title
        ttk.Label(
            parent,
            text=title,
            font=("Helvetica", 18, "bold")
        ).pack(pady=(15, 20))

        if HAS_MATPLOTLIB:
            self._build_plots(parent, result)
        else:
            ttk.Label(parent, text="Matplotlib is not available").pack()

        self._build_tables(parent, result)

    # =====================================================
    # -------------------- PLOTS --------------------------
    # =====================================================
    def _build_plots(self, parent: ttk.Frame, result: Merged_GA):
        gens = list(range(len(result)))
        gen_sizes = [g["gen_size"] for g in result]
        best_fitness = [g["best_chromosome"]["fitness"] for g in result]
        worst_fitness = [g["worst_chromosome"]["fitness"] for g in result]
        avg_fitness = [g["average_generations_fitness"] for g in result]
        ones_count = [sum(g["best_chromosome"]["bit_string"]) for g in result]

        # ----- Row frames for better layout -----
        row1 = ttk.Frame(parent)
        row1.pack(fill=BOTH, expand=True, pady=(0, 15))

        row2 = ttk.Frame(parent)
        row2.pack(fill=BOTH, expand=True, pady=(0, 15))

        row3 = ttk.Frame(parent)
        row3.pack(fill=BOTH, expand=True, pady=(0, 25))

        # ----- Plot 1: Generation Size -----
        fig1 = Figure(figsize=(12, 4), dpi=100)
        ax1 = fig1.add_subplot(111)
        ax1.plot(gens, gen_sizes, marker="o")
        ax1.set_title("Generation Size")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Size")
        ax1.grid(True, alpha=0.4)
        canvas1 = FigureCanvasTkAgg(fig1, master=row1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        # ----- Plot 2: Best & Worst Fitness -----
        fig2 = Figure(figsize=(12, 4), dpi=100)
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

        # ----- Plot 3: Average Fitness -----
        fig3 = Figure(figsize=(12, 4), dpi=100)
        ax3 = fig3.add_subplot(111)
        ax3.plot(gens, avg_fitness, marker="o", color="orange")
        ax3.set_title("Average Fitness Across Generations")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Average Fitness")
        ax3.grid(True, alpha=0.4)
        canvas3 = FigureCanvasTkAgg(fig3, master=row2)
        canvas3.draw()
        canvas3.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True, padx=5)

        # ----- Plot 4: Number of Selected Features -----
        fig4 = Figure(figsize=(12, 4), dpi=100)
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

        # ----- Plot 5: Best Fitness Stabilization -----
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
    def _build_tables(self, parent: ttk.Frame, result: Merged_GA):
        tables_frame = ttk.Frame(parent)
        tables_frame.pack(fill=BOTH, expand=True, padx=10, pady=15)

        # First table (Top 10 Best) - Full width
        table1_frame = ttk.Frame(tables_frame)
        table1_frame.pack(fill=BOTH, expand=True, pady=(0, 15))

        ttk.Label(table1_frame, text="Top 10 Best Chromosomes", font=("Helvetica", 12, "bold")).pack(pady=5)

        cols1 = ("Generation", "Fitness", "Selected Features", "Decoded Value")
        table1 = ttk.Treeview(table1_frame, columns=cols1, show="headings", height=15)
        for c in cols1:
            table1.heading(c, text=c)
            table1.column(c, anchor=CENTER)

        vsb1 = ttk.Scrollbar(table1_frame, orient=VERTICAL, command=table1.yview)
        table1.configure(yscrollcommand=vsb1.set)
        table1.pack(side=LEFT, fill=BOTH, expand=True)
        vsb1.pack(side=RIGHT, fill=Y)

        all_best = [(i, g["best_chromosome"]) for i, g in enumerate(result)]
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

        # Second table (Feature Usage) - Full width
        table2_frame = ttk.Frame(tables_frame)
        table2_frame.pack(fill=BOTH, expand=True)

        ttk.Label(table2_frame, text="Feature Usage (Best Chromosomes)", font=("Helvetica", 12, "bold")).pack(pady=5)

        table2 = ttk.Treeview(
            table2_frame,
            columns=("Feature Index", "Usage Count"),
            show="headings",
            height=15,
        )
        table2.heading("Feature Index", text="Feature Index")
        table2.heading("Usage Count", text="Usage Count")
        table2.column("Feature Index", anchor=CENTER)
        table2.column("Usage Count", anchor=CENTER)

        vsb2 = ttk.Scrollbar(table2_frame, orient=VERTICAL, command=table2.yview)
        table2.configure(yscrollcommand=vsb2.set)
        table2.pack(side=LEFT, fill=BOTH, expand=True)
        vsb2.pack(side=RIGHT, fill=Y)

        # Sort by usage count descending
        for idx, count in sorted(self._bit_index_statistics(result).items(), key=lambda x: x[1], reverse=True):
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
        return sum(bit_string)  # placeholder

    def _bit_index_statistics(self, result: Merged_GA) -> Dict[int, int]:
        stats: Dict[int, int] = {}
        for g in result:
            for i, v in enumerate(g["best_chromosome"]["bit_string"]):
                if v == 1:
                    stats[i] = stats.get(i, 0) + 1
        return dict(sorted(stats.items()))

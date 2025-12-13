import ttkbootstrap as ttk
import tkinter as tk

from interfaces.types import RunGAResult, Generations, Genome
from typing import Dict, List, Tuple

try:
    # Matplotlib for plotting within Tkinter
    import matplotlib
    matplotlib.use("Agg")  # use a non-interactive backend for headless environments
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


class ResultsPage(ttk.Frame):
    """Show GA results including a simple plot of best fitness per generation.

    If `results` is None, a mock RunGAResult is generated for demo purposes.
    """

    def __init__(self, master, results: RunGAResult | None = None):
        super().__init__(master)

        ttk.Label(self, text="Results", font=("Helvetica", 16, "bold")).pack(pady=12)

        # If no real results were supplied, build mock data
        if results is None:
            results = self._build_mock_result()

        # Show best-genome summary on the left
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ttk.Label(left, text="Best Genome", font=("Helvetica", 12, "bold")).pack(anchor="w")
        best = results.best_genome
        best_text = tk.Text(left, height=10, width=40)
        best_text.pack(padx=6, pady=6)
        best_text.insert("end", self._format_genome(best))
        best_text.configure(state="disabled")

        # Suggested features based on all generations (weighted by fitness)
        ttk.Label(left, text="Suggested Features", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(8, 0))
        suggested = self._suggest_best_features(results.generations, top_n=10)
        sug_text = tk.Text(left, height=8, width=40)
        sug_text.pack(padx=6, pady=6)
        for feat, score in suggested:
            sug_text.insert("end", f"{feat}: {score:.4f}\n")
        sug_text.configure(state="disabled")

        ttk.Button(left, text="Back to Start", command=master.show_start_page).pack(pady=6)

        # Plot area on the right
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ttk.Label(right, text="Fitness over Generations", font=("Helvetica", 12, "bold")).pack()

        if HAS_MATPLOTLIB:
            fig = Figure(figsize=(6, 3.6), dpi=100)
            ax = fig.add_subplot(111)
            gens, bests = self._best_fitness_by_generation(results.generations)
            _, avgs = self._average_fitness_by_generation(results.generations)

            ax.plot(gens, bests, marker="o", linestyle="-", color="#1f77b4", label="Best Fitness")
            ax.plot(gens, avgs, marker="s", linestyle="--", color="#ff7f0e", label="Average Fitness")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=right)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        else:
            ttk.Label(right, text="Matplotlib not available. Install matplotlib to see plots.").pack(padx=6, pady=20)

    def _format_genome(self, genome: Genome) -> str:
        lines = [f"id: {genome.get('id')}", f"fitness: {genome.get('fitness'):.4f}", f"accuracy: {genome.get('accuracy'):.4f}", "features:"]
        for f in genome.get("features", []):
            lines.append(f"  - {f}")
        return "\n".join(lines)

    def _best_fitness_by_generation(self, generations: Generations) -> tuple[List[int], List[float]]:
        gens = []
        bests = []
        for g in sorted(generations.keys()):
            gen_list = generations[g]
            best_fit = max((item.get("fitness", 0.0) for item in gen_list), default=0.0)
            gens.append(g)
            bests.append(best_fit)
        return gens, bests

    def _average_fitness_by_generation(self, generations: Generations) -> tuple[List[int], List[float]]:
        gens = []
        avgs = []
        for g in sorted(generations.keys()):
            gen_list = generations[g]
            if not gen_list:
                avg = 0.0
            else:
                total = sum((item.get("fitness", 0.0) for item in gen_list))
                avg = total / len(gen_list)
            gens.append(g)
            avgs.append(avg)
        return gens, avgs

    def _suggest_best_features(self, generations: Generations, top_n: int = 10) -> List[Tuple[str, float]]:
        """Return top features ranked by weighted fitness across all generations.

        We weight each occurrence of a feature by the genome's fitness so that
        frequently high-fitness features bubble to the top. Returns a list of
        (feature, score) tuples sorted descending by score.
        """
        scores: Dict[str, float] = {}
        for gen_list in generations.values():
            for genome in gen_list:
                fitness = float(genome.get("fitness", 0.0) or 0.0)
                for feat in genome.get("features", []):
                    scores[feat] = scores.get(feat, 0.0) + fitness

        sorted_feats = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_feats[:top_n]

    def _build_mock_result(self) -> RunGAResult:
        """Return a mock RunGAResult instance compatible with the project's types.

        This mirrors the shape produced by the project's `MockGAService` but is
        kept local so the results page can be previewed in isolation.
        """
        import random

        generations: Generations = {}
        num_generations = 8
        for gen in range(num_generations):
            genomes = []
            for i in range(6):
                genome = {
                    "parent_id": random.randint(0, 5),
                    "id": gen * 100 + i,
                    "fitness": random.uniform(0.0, 1.0),
                    "accuracy": random.uniform(0.6, 0.98),
                    "features": [f"feature_{k}" for k in random.sample(range(30), random.randint(4, 12))],
                }
                genomes.append(genome)
            generations[gen] = genomes

        last_gen = generations[num_generations - 1]
        best_genome = max(last_gen, key=lambda g: g["fitness"])

        result = RunGAResult()
        result.best_genome = best_genome
        result.generations = generations
        return result

    # Note: input-data viewing removed — this page no longer shows AppContext data.
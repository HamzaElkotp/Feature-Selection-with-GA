from interfaces.api_contract import GAInterface
from interfaces.types import *
from typing import List, Optional, Callable
import random
import time


class MockGAService(GAInterface):

    def run_ga(self, parms: RunGAParameters, on_complete: Optional[Callable[[RunGAResult], None]] = None) -> None:
        """
        Mock implementation of the GA algorithm.
        Calls `on_complete(result)` when finished if the callback is provided.
        """

        # Simulate computation time (optional)
        time.sleep(0.3)

        # ---- Generate fake generations ----
        generations: Generations = {}
        num_generations = 10   # fixed for the mock

        for gen in range(num_generations):
            genomes: List[Genome] = []

            for i in range(5):  # 5 genomes per generation
                genome: Genome = {
                    "parent_id": random.randint(0, 5),
                    "id": gen * 10 + i,
                    "fitness": random.uniform(0, 1),
                    "accuracy": random.uniform(0.6, 0.95),
                    "features": [
                        f"feature_{k}"
                        for k in random.sample(range(20), random.randint(5, 10))
                    ],
                }
                genomes.append(genome)

            generations[gen] = genomes

        # ---- Pick the "best" genome from the last generation ----
        last_gen = generations[num_generations - 1]
        best_genome = max(last_gen, key=lambda g: g["fitness"])

        # ---- Build a RunGAResult ----
        result = RunGAResult()
        result.best_genome = best_genome
        result.generations = generations

        # Signal completion via callback if provided
        if on_complete:
            try:
                on_complete(result)
            except Exception:
                # Swallow exceptions from callback to avoid crashing the GA thread
                pass

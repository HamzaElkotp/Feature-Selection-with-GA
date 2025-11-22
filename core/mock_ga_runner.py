from interfaces.api_contract import GAInterface
from interfaces.types import *
from typing import List
import random
import time


class MockGAService(GAInterface):

    def run_ga(self, parms: RunGAParameters) -> RunGAResult:
        """
        Mock implementation of the GA algorithm.
        Returns dummy data so the GUI can be developed and tested.
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

        return result

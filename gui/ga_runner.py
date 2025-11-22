# core/ga_runner.py
import time
from shared.utils import Observable

class GeneticAlgorithm(Observable):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.best_solution = None

    def run(self):
        # Simulate a long-running GA process
        for generation in range(1, 11):
            time.sleep(0.5)  # simulate computation
            fitness = 0.1 * generation
            self.notify("progress", {"generation": generation, "fitness": fitness})

        # Once done
        result = {"best_solution": [1, 0, 1, 1], "fitness": 0.93}
        self.notify("complete", result)
        return result
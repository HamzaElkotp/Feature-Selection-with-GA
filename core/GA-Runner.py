import os
from typing import TypedDict

import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

from cleaned import (
    initialize_population,
    validate_bitstring_chromosome,
    compute_fitness # can use compute_fitness_list better
)

class Chromosome(TypedDict):
    bit_string: np.int64
    fitness: float

class Generation(TypedDict):
    average_fitness: float
    best_chromosome: Chromosome
    worst_chromosome: Chromosome


class GA:
    def __init__(
        self,
        initiate_population,
        validate_chromosome,
        elitism,
        selection,
        crossover,
        mutation,
        computer_generation_fitness,
        num_generations=100,
        population_size=100,
        mutation_percent=0.05,
        elitism_percent=0.05,
        alpha=1,
        beta=1,
    ):
        self.initiate_population = initiate_population
        self.validate_chromosome = validate_chromosome
        self.elitism = elitism
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.computer_generation_fitness = computer_generation_fitness

        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_percent = mutation_percent
        self.elitism_percent = elitism_percent
        self.alpha = alpha
        self.beta = beta

    def run(self):
        # Store all generations
        generations:[Generation] = []


        population = self.initiate_population(self.population_size)
        # computer fitness first
        # last_generated_population = population # always will equal to the generation that = a sorted list of chromosomes class
        # extract info
        # init_Generation = Generation()
        generations.append(population)

        for i in range(self.num_generations):
            # elites = self.elitism(population, self.elitism_percent)

            # selected = self.selection(population)
            # offspring = self.crossover(selected)
            #
            # mutated = self.mutation(offspring)

            # population = mutated
            new_gen = Generation()
            generations.append(new_gen)

        return generations

    def master_run(self, num_runs, max_workers=min(32, os.cpu_count() + 4)):
        all_results = [None] * num_runs

        def run_wrapper(i):
            # Runs one GA execution and returns (index, result)
            res = self.run()
            return i, res

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_wrapper, i) for i in range(num_runs)]

            for future in as_completed(futures):
                i, result = future.result()
                all_results[i] = result

        return all_results


# MyGa = GA(
#     initiate_population = initialize_population,
#     validate_chromosome = validate_bitstring_chromosome,
#     elitism = my_elitism,
#     selection = my_selection,
#     crossover = my_crossover,
#     mutation = my_mutation,
#     computer_generation_fitness = compute_fitness, ######### replace with compute_fitness_list
#     num_generations=100,
#     population_size=20,
#     mutation_percent=0.09,
#     elitism_percent=0.1,
#     alpha=0.8,
#     beta=0.2,
# )
#
# MyGa.run()
# MyGa.master_run(num_runs=30)
import math
import os
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from GA_functions import (
# Types
Chromosome,
Population,
Generation,

# Helper Functions
validated_inputs,
extract_gen_info,
unique_list,
unique_population,
Descending_order_fitnesses,


# GA General Functions
create_bitstring_chromosome,
validate_bitstring_chromosome,
initialize_population,
get_population_fitness,

# Selection Methods
random_selection_unique,
roulette_wheel_selection,
tournament_selection,

# Cross-over Methods
population_k_point_crossover,

# Elitism Methods
elitism_selector,

# Mutation Methods
bit_flip_mutator,
complement_mutator,
reverse_mutator,
rotation_mutator,
)


class GA:
    def __init__(
        self,
        initiate_population,
        elitism,
        selection,
        crossover,
        mutation,
        compute_generation_fitness,

        dataset_path,
        result_col_name,

        num_generations=100,
        population_size=20,
        crossover_k_points=1,
        mutation_percent=1,
        elitism_percent=1,
        alpha=1,
        beta=1,
    ):
        self.initiate_population = initiate_population
        self.elitism = elitism
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.compute_generation_fitness = compute_generation_fitness

        self.num_generations = num_generations
        self.population_size = population_size
        self.crossover_k_points = crossover_k_points ###
        self.mutation_percent = mutation_percent
        self.elitism_percent = elitism_percent
        self.alpha = alpha
        self.beta = beta

        self.dataset_path = dataset_path
        self.result_col_name = result_col_name
        self.prediction_target = None
        self.features = None
        self.num_features=0

    def initiate_dataset(self):
        df = pd.read_csv(self.dataset_path)
        self.prediction_target = df[self.result_col_name]
        self.features = df.drop(columns=[self.result_col_name])
        self.num_features = len(self.features.columns)

    def run(self):
        self.initiate_dataset()

        # validated_inputs(self.dataset_path, self.result_col_name, self.initiate_population, self.features, self.prediction_target)

        # Store all generations
        generations:[Generation] = []

        population = self.initiate_population(self.population_size, self.num_features)
        last_generated_population:Population = self.compute_generation_fitness(population, self.features, self.prediction_target, alpha=self.alpha, beta=self.beta)
        last_generated_population = Descending_order_fitnesses(last_generated_population)

        generations.append(extract_gen_info(last_generated_population))

        for i in range(self.num_generations):
            # Determine number of parents to select to generate new size that is x1.5
            num_of_combinations_needed = math.ceil(len(last_generated_population) * 1.5)
            num_of_combinations_needed -= num_of_combinations_needed%2
            if(num_of_combinations_needed <= 0):
                num_of_combinations_needed = len(last_generated_population)
                num_of_combinations_needed -= num_of_combinations_needed % 2

            # Select parents from old generation
            selected_parents: Population = self.selection(last_generated_population, num_of_combinations_needed)
            # Marriage parents with each others and get unique children
            new_children = self.crossover(selected_parents, self.crossover_k_points)

            # Do mutation from old generation and push them to the new
            mutation_number = max(math.ceil(len(last_generated_population) * self.mutation_percent / 100), 1)
            mutated = self.mutation(last_generated_population, mutation_number)
            new_children.extend(mutated)

            # Make sure new generation children are all unique (Optimization before compute fitness)
            new_children = unique_list(new_children)

            # Compute fitness of children
            new_children_with_fitness: Population = self.compute_generation_fitness(new_children, self.features, self.prediction_target, alpha=self.alpha, beta=self.beta)

            # Do elitism from old generation and push them to the new
            elitism_number = max(math.ceil(len(last_generated_population) * self.elitism_percent / 100), 1)
            elites = self.elitism(last_generated_population, elitism_number)
            new_children_with_fitness.extend(elites)

            # Make sure new generation children are all unique
            new_children_with_fitness = unique_population(new_children_with_fitness)

            # Sort the new Generation
            new_children_with_fitness = Descending_order_fitnesses(new_children_with_fitness)

            # Store Generation
            new_gen = extract_gen_info(new_children_with_fitness)
            generations.append(new_gen)

            last_generated_population = new_children_with_fitness

        return generations


    def master_run(self, num_runs, max_workers=min(32, os.cpu_count())):
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


MyGa = GA(
initiate_population=initialize_population,
elitism=elitism_selector,
selection=tournament_selection,
crossover=population_k_point_crossover,
mutation=complement_mutator,
compute_generation_fitness=get_population_fitness,
dataset_path=r"D:\SelfAcademicLearn\University\Year 3\AI\Project\processed_Breast_Cancer_Dataset.csv",
result_col_name="diagnosis",
num_generations=10,
population_size=50,
crossover_k_points=2,
mutation_percent=4,
elitism_percent=1,
alpha=1,
beta=1,
)
#
x:[Generation] = MyGa.run()
print(x)
# MyGa.master_run(num_runs=100)
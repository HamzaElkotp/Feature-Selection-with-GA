from core.GA_Runner import GA
from interfaces.api_contract import GAInterface
from interfaces.types import *
from typing import List, Optional, Callable
import random
import time

from core.GA_functions import (
# Types
Chromosome,
Population,
Generation,
Merged_Generation,
Merged_GA,

# Helper Functions
validated_inputs,
extract_gen_info,
unique_list,
unique_population,
Descending_order_fitnesses,
merge_GAs,

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


class GA_Service(GAInterface):

    def run_ga(self, params: RunGAParameters, on_complete: Optional[Callable[[Merged_GA, Merged_GA], None]] = None) -> None:
        new_Ga = GA(
            initiate_population=initialize_population, # function call
            elitism=elitism_selector, # function call
            selection = (
                tournament_selection
                if params.ga_parameters.selection_method == SelectionMethod.TOURNAMENT
                else roulette_wheel_selection
                if params.ga_parameters.selection_method == SelectionMethod.ROULETTE
                else random_selection_unique
            ), # function call
            crossover=population_k_point_crossover, # function call
            mutation= (
                bit_flip_mutator
                if params.ga_parameters.mutation_method == MutationMethod.Bit_Flip
                else complement_mutator
                if params.ga_parameters.mutation_method == MutationMethod.Bit_String_Complement
                else reverse_mutator
                if params.ga_parameters.mutation_method == MutationMethod.Bit_String_Reverse
                else rotation_mutator
            ), # function call
            compute_generation_fitness=get_population_fitness, # function call
            dataset_path=params.dataset_file_path,
            result_col_name=params.ga_parameters.result_col_name,
            num_generations=params.ga_parameters.num_of_generations,
            population_size=params.ga_parameters.initial_population_size,
            crossover_k_points=params.ga_parameters.crossover_k_points,
            mutation_percent=params.ga_parameters.mutation_percent,
            elitism_percent=params.ga_parameters.elitism_percent,
            alpha=params.ga_parameters.alpha,
            beta=params.ga_parameters.beta,
        )

        dt_result:Merged_GA = new_Ga.master_run(num_runs=1) # type: ignore
        rf_result: Merged_GA = new_Ga.master_run(num_runs=1) # type: ignore

        # Signal completion via callback if provided
        if on_complete:
            try:
                on_complete(dt_result, rf_result)
            except Exception:
                pass

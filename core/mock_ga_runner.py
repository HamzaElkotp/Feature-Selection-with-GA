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


class MockGAService(GAInterface):

    def run_ga(self, parms: RunGAParameters, on_complete: Optional[Callable[[Merged_GA], None]] = None) -> None:
        new_Ga = GA(
            initiate_population=initialize_population,
            elitism=elitism_selector,
            selection=tournament_selection,
            crossover=population_k_point_crossover,
            mutation=complement_mutator,
            compute_generation_fitness=get_population_fitness,
            dataset_path=r"D:\SelfAcademicLearn\University\Year 3\AI\Project\processed_Breast_Cancer_Dataset.csv",
            result_col_name="diagnosis",
            num_generations=4,
            population_size=20,
            crossover_k_points=2,
            mutation_percent=4,
            elitism_percent=1,
            alpha=1,
            beta=1,
        )

        result:Merged_GA = new_Ga.master_run(num_runs=20) # type: ignore

        print(result)

        # Signal completion via callback if provided
        if on_complete:
            try:
                on_complete(result)
            except Exception:
                pass

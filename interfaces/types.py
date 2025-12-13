from dataclasses import dataclass
from typing import TypedDict, List, Dict, TypeAlias

from interfaces.enums import *

@dataclass
class GAParameters:
    elitism_percent: int
    mutation_percent: int
    crossover_k_points: int  # CrossoverMethod
    initial_population_size: int

    alpha: float
    beta: float
    num_of_generations: int

    # dataset_path: str
    result_col_name: str

    selection_method: SelectionMethod
    mutation_method: MutationMethod


@dataclass
class RunGAParameters:
    dataset_file_path: str
    mode: RunMode
    ga_parameters: GAParameters

class Genome(TypedDict):
    parent_id: int
    id: int
    fitness: float
    accuracy: float
    features: List[str]

Generations: TypeAlias = Dict[int, List[Genome]]

class RunGAResult:
    best_genome: Genome
    generations: Generations

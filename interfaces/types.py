from dataclasses import dataclass
from typing import TypedDict, List, Dict, TypeAlias

from interfaces.enums import *

@dataclass
class GAParameters:
    selection_method: SelectionMethod
    crossover_method: CrossoverMethod
    mutation_method: MutationMethod
    termination_condition: TerminationCondition
    termination_condition_n: float
    elitism_percent: float
    mutation_percent: float
    alpha: float
    beta: float

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

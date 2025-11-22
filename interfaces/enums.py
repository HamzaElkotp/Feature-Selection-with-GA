from enum import Enum, auto

class RunMode(Enum):
    DIM = auto() # Deterministic Input Mode
    EBM = auto() # Empirical Benchmarking Mode

class SelectionMethod(Enum):
    ROULETTE = auto()
    RANKING = auto()
    TOURNAMENT = auto()

class CrossoverMethod(Enum):
    SINGLE_POINT = auto()
    MULTI_POINT = auto()

class MutationMethod(Enum):
    BIT_FLIP = auto()
    INVERSION = auto()

class TerminationCondition(Enum):
    AFTER_N_GENERATIONS = auto()
    AFTER_N_SECONDS = auto()
    AFTER_FITNESS_REACHES_N = auto()
    NO_IMPROVEMENT_SINCE_N_GENERATIONS = auto()


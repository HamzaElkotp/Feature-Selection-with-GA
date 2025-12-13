from enum import Enum, auto

class RunMode(Enum):
    DIM = auto() # Deterministic Input Mode
    EBM = auto() # Empirical Benchmarking Mode

class SelectionMethod(Enum):
    ROULETTE = auto()
    TOURNAMENT = auto()
    RANDOM = auto()

# class CrossoverMethod(Enum):
#     SINGLE_POINT = auto()
#     MULTI_POINT = auto()

class MutationMethod(Enum):
    Bit_Flip = auto()
    Bit_String_Complement = auto()
    Bit_String_Reverse = auto()
    Bit_String_Rotation = auto()

# class TerminationCondition(Enum):
#     AFTER_N_GENERATIONS = auto()
#     AFTER_N_SECONDS = auto()
#     AFTER_FITNESS_REACHES_N = auto()
#     NO_IMPROVEMENT_SINCE_N_GENERATIONS = auto()


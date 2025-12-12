from re import S
import numpy as np
import random
import array
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from typing import TypedDict, List

class Chromosome(TypedDict):
    bit_string: np.int64
    fitness: float

Population = List[Chromosome]

class Generation(TypedDict):
    average_fitness: float
    total_fitness: float
    best_chromosome: Chromosome
    worst_chromosome: Chromosome



def create_bitstring_chromosome(num_features: int) :
    return tuple(np.random.randint(2, size=num_features))

def validate_bitstring_chromosome(chromosome:tuple) -> bool :
    selected_indices = chromosome.count(1)
    if selected_indices == 0:
        return False
    return True

def initialize_population(population_size: int , num_features: int):
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if population_size >= pow(2, num_features):
        raise ValueError("population_size must be less than 2^num_features.")
    if num_features < 2:
        raise ValueError("num_features must be at least 2")

    population = set()
    while len(population) < population_size:
        chromosome = create_bitstring_chromosome(num_features)
        if validate_bitstring_chromosome(chromosome):
            population.add(chromosome)
    return [np.array(c) for c in population]


"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""
# Example
# population = initialize_population(population_size=10, num_features=4)
# for chromo in population:
#     print(chromo)
"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""
FITNESS FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""

def compute_accuracy(chromosome, dataset_features, prediction_target) -> np.floating:
    selected_indices = np.where(chromosome == 1)[0]
    dataset_features_selected = dataset_features.iloc[:, selected_indices]
    model = DecisionTreeClassifier()
    accuracy = np.mean(cross_val_score(model, dataset_features_selected, prediction_target, cv=3))

    return accuracy


def compute_fitness(chromosome, dataset_features, prediction_target, alpha=1, beta=1) -> float:
    selected_indices = np.where(chromosome == 1)[0]
    accuracy = compute_accuracy(chromosome, dataset_features, prediction_target)

    features_total = len(chromosome)
    features_selected = len(selected_indices)

    penalty = beta * (features_selected / features_total)
    reward = alpha * accuracy

    fitness_value:float = float(reward - penalty)
    return fitness_value


def get_population_fitness(_population, dataset_features, prediction_target, alpha=1, beta=1) -> Population:
    population_with_fitness:Population = []

    for chrom in _population:
        fitness = compute_fitness(chrom, dataset_features, prediction_target, alpha, beta)
        chrom = Chromosome(bit_string=chrom, fitness=fitness)
        population_with_fitness.append(chrom)

    return population_with_fitness


def Descending_order_fitnesses(population_with_fitness:Population) -> Population:
    return sorted(
        population_with_fitness,
        key=lambda chrom: chrom["fitness"],
        reverse=True
    )


"""""""""""""""""""""""""""""""""""""""""
SELECTION FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""

"""
    FOR RANDOM SELECTION
"""
def random_selection(population: Population):
    return random.choice(population)

def random_selection_unique(population: Population, k=2):
    return random.sample(population, k)


"""
    FOR ROULETTE WHEEL SELECTION
"""

def shift_fitnesses(sorted_population: Population):
    min_fit = min(chromo["fitness"] for chromo in sorted_population)
    if min_fit < 0:
        shift = -min_fit
    else:
        shift = 0

    return [chromo["fitness"] + shift for chromo in sorted_population]

def Descending_order_ratios(shifted_fitness):
    total_fitness = sum(shifted_fitness)

    ratios = [(fitness / total_fitness) * 100 for fitness in shifted_fitness]

    return ratios


def roulette_wheel(ratios):
    roulette = []
    cumulative_sum = 0

    for r in ratios:
        cumulative_sum += r
        roulette.append(cumulative_sum)

    return roulette


def roulette_wheel_selection(ratio_list, population):
    r = random.uniform(0, ratio_list[-1])
    for i in range(len(ratio_list)):
        if r <= ratio_list[i] :
            return population[i]


"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""
# Example
x = initialize_population(20, 10)
print(x)

g = get_population_fitness(x, 0, 0)
print(g)
print(g[0]["bit_string"])

sorted_pop = Descending_order_fitnesses(g)
print(sorted_pop)

shifted_fitness = shift_fitnesses(sorted_pop)
print(shifted_fitness)

ratios = Descending_order_ratios(shifted_fitness)
print(ratios)

ratio_list = roulette_wheel(ratios)
print(ratio_list)

selected = roulette_wheel_selection(ratio_list, sorted_pop)
print(selected)


"""
    FOR TOURNAMENT SELECTION
"""
def tournament_selection(population, k=3):
    # sample k individuals without replacement
    contestants = random.sample(population, k)

    winner = max(contestants, key=lambda chrom: chrom["fitness"])

    return winner



"""""""""""""""""""""""""""""""""""""""""
CROSS-OVER FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""

def k_points_crossover(parent1, parent2, k):
    chromo_len = len(parent1)
    points = random.sample(range(1, chromo_len - 1), k)
    points.sort()
    points = [0] + points + [chromo_len]

    child1 = []
    child2 = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if i % 2 == 0:  # takes even segments: p1 → child1, p2 → child2
            child1 += parent1[start:end]
            child2 += parent2[start:end]
        else:  # takes odd segment: p2 → child1, p1 → child2
            child1 += parent2[start:end]
            child2 += parent1[start:end]

    return child1, child2


"""""""""""""""""""""""""""""""""""""""""
MUTATION FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""
#bitflip
#c : 1001
#c flip : 1101
def bit_flip_mutation(chromo, mutation_rate=0.1):
    # Pick a random gene index
    r = random.randrange(len(chromo))
    if np.random.rand() < mutation_rate:
        if chromo[r] == 1 :
             chromo[r] = 0
        else :
            chromo[r] = 1
    return chromo

#Complement
#c : 0101
#c comp : 1010
def Complement_mutation(chromo, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        for i in range(len(chromo)) :
            if chromo[i] == 1 :
                chromo[i] = 0
            else :
                chromo[i] = 1
    return chromo

#reverse
#c : 1011
#c rev : 1101
def reverse_mutation(chromo, mutation_rate=0.1):
    temp_offspring1 = np.zeros(len(chromo), dtype=int) # create a 0 chromosome .
    if np.random.rand() < mutation_rate:
        for i in range(len(chromo)) :
            temp_offspring1[i] = chromo[len((chromo)-1)-i]
    offspring1 = temp_offspring1
    return offspring1

# Rotation
#c: 1011
#c Rotation : 1110
def Rotation_mutation(chromo, mutation_rate=0.1):
    # Pick a random gene index
    r = random.randint(1, len(chromo) - 1)
    if np.random.rand() < mutation_rate:
        # numpy concatenation
        temp_offspring = np.concatenate((chromo[r:], chromo[:r]))
    else:
        temp_offspring = chromo

    offspring = temp_offspring
    return offspring
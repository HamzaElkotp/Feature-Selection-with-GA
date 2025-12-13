from re import S
import numpy as np
import random
import array
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from typing import TypedDict, List


class Chromosome(TypedDict):
    bit_string: List[int]
    fitness: float

Population = List[Chromosome]

class Generation(TypedDict):
    average_fitness: float
    total_fitness: float
    best_chromosome: Chromosome
    worst_chromosome: Chromosome
    gen_size: int


def create_bitstring_chromosome(num_features: int) :
    return tuple(int(x) for x in np.random.randint(2, size=num_features))

def validate_bitstring_chromosome(chromosome:tuple) -> bool :
    selected_indices = chromosome.count(1)
    if selected_indices == 0:
        return False
    return True

def initialize_population(population_size: int , num_features: int):
    population = set()
    while len(population) < population_size:
        chromosome = create_bitstring_chromosome(num_features)
        if validate_bitstring_chromosome(chromosome):
            population.add(chromosome)

    return [list(c) for c in population]


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
    # Decode chromosome to features
    selected_indices = np.where(np.array(chromosome) == 1)[0]
    dataset_features_selected = dataset_features.iloc[:, selected_indices]
    model = DecisionTreeClassifier()
    accuracy = np.mean(cross_val_score(model, dataset_features_selected, prediction_target, cv=3))

    return accuracy


# (alpha * accuracy) - (beta * selected / tootal feature)
def compute_fitness(chromosome, dataset_features, prediction_target, alpha=1, beta=1) -> float:
    selected_indices = np.where(np.array(chromosome) == 1)[0]
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

def random_selection_unique(population: Population, parents_needed=2):

    selected_list = []
    last_selected = None

    while len(selected_list) < parents_needed:
        selected = random_selection(population)

        if selected == last_selected:
            continue  # reject and resample

        selected_list.append(selected)
        last_selected = selected

    return selected_list

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


def roulette_wheel_selector(ratio_list, population: Population)-> Chromosome:
    r = random.uniform(0, ratio_list[-1])
    for i in range(len(ratio_list)):
        if r <= ratio_list[i] :
            return population[i]

def roulette_wheel_selection(population: Population, parents_needed)-> Population:
    shifted_fitness = shift_fitnesses(population)
    ratios = Descending_order_ratios(shifted_fitness)
    ratio_list = roulette_wheel(ratios)

    selected_list = []
    last_selected = None

    while len(selected_list) < parents_needed:
        selected = roulette_wheel_selector(ratio_list, population)

        if selected == last_selected:
            continue  # reject and resample

        selected_list.append(selected)
        last_selected = selected

    return selected_list



"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""
# Example
# x = initialize_population(20, 10)
# print(x)
#
# g = get_population_fitness(x, 0, 0)
# print(g)
# print(g[0]["bit_string"])
#
# sorted_pop = Descending_order_fitnesses(g)
# print(sorted_pop)
#
# shifted_fitness = shift_fitnesses(sorted_pop)
# print(shifted_fitness)
#
# ratios = Descending_order_ratios(shifted_fitness)
# print(ratios)
#
# ratio_list = roulette_wheel(ratios)
# print(ratio_list)
#
# selected = roulette_wheel_selector(ratio_list, sorted_pop)
# print(selected)


"""
    FOR TOURNAMENT SELECTION
"""
def tournament_selector(population: Population, k=3) -> Chromosome:
    # sample k individuals without replacement
    contestants = random.sample(population, k)

    winner = max(contestants, key=lambda chrom: chrom["fitness"])

    return winner


def tournament_selection(population: Population, parents_needed) -> Population:
    selected_list = []
    last_selected = None

    while len(selected_list) < parents_needed:
        selected = tournament_selector(population)

        if selected == last_selected:
            continue  # reject and resample

        selected_list.append(selected)
        last_selected = selected

    return selected_list



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

def population_k_point_crossover(population: Population, k): # do cross-over and return unique children
    new_children = []
    seen = set()

    for i in range(1, len(population), 2):
        child1, child2 = k_points_crossover(
            population[i]["bit_string"],
            population[i - 1]["bit_string"],
            k
        )

        if validate_bitstring_chromosome(child1):
            key = tuple(child1)
            if key not in seen:
                seen.add(key)
                new_children.append(child1)

        if validate_bitstring_chromosome(child2):
            key = tuple(child2)
            if key not in seen:
                seen.add(key)
                new_children.append(child2)

    return new_children


"""""""""""""""""""""""""""""""""""""""""
Elitism FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""
def elitism_selector(population: Population, k: int) -> Population:
    selected: Population = []

    for i in range(k):
        selected.append(population[i])

    return selected


"""""""""""""""""""""""""""""""""""""""""
MUTATION FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""
#bitflip
#c : 1001
#c flip : 1101
def bit_flip_mutation(chromo):
    r = random.randrange(len(chromo))
    if chromo[r] == 1 :
         chromo[r] = 0
    else :
        chromo[r] = 1
    return chromo

def bit_flip_mutator(population: Population, k: int):
    new_chromos = []
    for i in range(k):
        new_chromos.append(bit_flip_mutation(population[i]["bit_string"]))
    return new_chromos

#Complement
#c : 0101
#c comp : 1010
def Complement_mutation(chromo):
    for i in range(len(chromo)) :
        if chromo[i] == 1 :
            chromo[i] = 0
        else :
            chromo[i] = 1
    return chromo

def complement_mutator(population: Population, k: int):
    new_chromos = []
    for i in range(k):
        new_chromos.append(Complement_mutation(population[i]["bit_string"]))
    return new_chromos

#reverse
#c : 1011
#c rev : 1101
def reverse_mutation(chromo):
    temp_offspring1 = np.zeros(len(chromo), dtype=int) # create a 0 chromosome .
    for i in range(len(chromo)) :
        temp_offspring1[i] = chromo[len((chromo)-1)-i]
    offspring1 = temp_offspring1
    return offspring1

def reverse_mutator(population: Population, k: int):
    new_chromos = []
    for i in range(k):
        new_chromos.append(reverse_mutation(population[i]["bit_string"]))
    return new_chromos

# Rotation
#c: 1011
#c Rotation : 1110
def Rotation_mutation(chromo):
    # Pick a random gene index
    r = random.randint(1, len(chromo) - 1)
    # numpy concatenation
    temp_offspring = np.concatenate((chromo[r:], chromo[:r]))

    offspring = temp_offspring
    return offspring

def rotation_mutator(population: Population, k: int):
    new_chromos = []
    for i in range(k):
        new_chromos.append(Rotation_mutation(population[i]["bit_string"]))
    return new_chromos



"""""""""""""""""""""""""""""""""""""""""
Helper FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""
def unique_list(aList):
    unique_items = []

    for item in aList:
        if item not in unique_items:
            unique_items.append(item)

    return unique_items

def unique_population(population: Population) -> Population:
    seen = set()
    unique:Population = []

    for chrom in population:
        key = tuple(chrom["bit_string"])  # genotype identity

        if key not in seen:
            seen.add(key)
            unique.append(chrom)

    return unique

def validated_inputs(dataset_path, res_col_name, population_size, features, target):
    if len(dataset_path) == 0:
        raise ValueError("Dataset Path cannot be empty")

    if len(res_col_name) == 0:
        raise ValueError("DataResult Column Name cannot be empty")

    if len(features) == 0:
        raise ValueError("Features cannot be empty")

    if len(target) == 0:
        raise ValueError("Features cannot be empty")

    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if population_size >= pow(2, len(features)):
        raise ValueError("population_size must be less than 2^num_features.")
    if len(features) < 2:
        raise ValueError("num_features must be at least 2")


def extract_gen_info(_population:Population) -> Generation:
    total = sum([chromo["fitness"] for chromo in _population])
    sz = len(_population)
    tmp_gen: Generation = {
        "average_fitness": total/sz,
        "total_fitness": total,
        "best_chromosome": _population[0],
        "worst_chromosome": _population[-1],
        "gen_size": sz
    }
    return tmp_gen
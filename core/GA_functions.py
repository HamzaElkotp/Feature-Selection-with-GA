from re import S
import numpy as np
import random
import array
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from typing import TypedDict, List
from config.settings import GPU_AVAILABLE, USE_GPU

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
except Exception:
    CUPY_AVAILABLE = False
    cp = None


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

class Merged_Generation(TypedDict):
    best_chromosome: Chromosome
    worst_chromosome: Chromosome
    gen_size: int
    total_generations_fitness: float
    average_generations_fitness: float

Merged_GA = List[Merged_Generation]




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


def get_population_fitness(_population, dataset_features, prediction_target, alpha=1, beta=1, use_gpu=None) -> Population:
    """
    Compute fitness for entire population.
    If GPU is available and enabled, uses GPU-accelerated batch processing.
    """
    if use_gpu is None:
        use_gpu = USE_GPU and GPU_AVAILABLE and CUPY_AVAILABLE
    
    if use_gpu:
        return get_population_fitness_gpu(_population, dataset_features, prediction_target, alpha, beta)
    else:
        return get_population_fitness_cpu(_population, dataset_features, prediction_target, alpha, beta)


def get_population_fitness_cpu(_population, dataset_features, prediction_target, alpha=1, beta=1) -> Population:
    """CPU-based fitness computation (original implementation)."""
    population_with_fitness:Population = []

    for chrom in _population:
        fitness = compute_fitness(chrom, dataset_features, prediction_target, alpha, beta)
        chrom = Chromosome(bit_string=chrom, fitness=fitness)
        population_with_fitness.append(chrom)

    return population_with_fitness


def get_population_fitness_gpu(_population, dataset_features, prediction_target, alpha=1, beta=1) -> Population:
    """
    GPU-accelerated fitness computation using CuPy for batch processing.
    Note: ML models still run on CPU, but array operations are GPU-accelerated.
    """
    if cp is None:
        # Fallback to CPU if CuPy is not available
        return get_population_fitness_cpu(_population, dataset_features, prediction_target, alpha, beta)
    
    population_with_fitness:Population = []
    
    # Convert dataset to GPU arrays for faster indexing
    try:
        features_array_gpu = cp.asarray(dataset_features.values, dtype=cp.float32)
        target_array_gpu = cp.asarray(prediction_target.values, dtype=cp.float32)
    except Exception:
        # If GPU operations fail, fall back to CPU
        return get_population_fitness_cpu(_population, dataset_features, prediction_target, alpha, beta)
    
    # Process in batches for better GPU utilization
    batch_size = min(32, len(_population))  # Process 32 chromosomes at a time
    
    for i in range(0, len(_population), batch_size):
        batch = _population[i:i+batch_size]
        
        for chrom in batch:
            # Use GPU for array operations
            chrom_gpu = cp.asarray(chrom, dtype=cp.int32)
            selected_indices = cp.where(chrom_gpu == 1)[0]
            
            if len(selected_indices) == 0:
                # Invalid chromosome, assign low fitness
                fitness = -1.0
            else:
                # Convert back to CPU for sklearn (sklearn doesn't support GPU)
                selected_indices_cpu = cp.asnumpy(selected_indices)
                dataset_features_selected = dataset_features.iloc[:, selected_indices_cpu]
                
                # Compute accuracy (still CPU-based for sklearn)
                model = DecisionTreeClassifier()
                accuracy = np.mean(cross_val_score(model, dataset_features_selected, prediction_target, cv=3))
                
                # Compute fitness
                features_total = len(chrom)
                features_selected = len(selected_indices_cpu)
                penalty = beta * (features_selected / features_total)
                reward = alpha * accuracy
                fitness = float(reward - penalty)
            
            chrom_dict = Chromosome(bit_string=chrom, fitness=fitness)
            population_with_fitness.append(chrom_dict)
    
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

def population_k_point_crossover(population: Population, k, use_gpu=None): # do cross-over and return unique children
    """Crossover with optional GPU acceleration."""
    if use_gpu is None:
        use_gpu = USE_GPU and GPU_AVAILABLE and CUPY_AVAILABLE
    
    if use_gpu and len(population) > 10:  # Use GPU for larger populations
        return population_k_point_crossover_gpu(population, k)
    else:
        return population_k_point_crossover_cpu(population, k)


def population_k_point_crossover_cpu(population: Population, k): # do cross-over and return unique children
    """CPU-based crossover (original implementation)."""
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


def population_k_point_crossover_gpu(population: Population, k):
    """GPU-accelerated batch crossover using CuPy."""
    if cp is None:
        # Fallback to CPU if CuPy is not available
        return population_k_point_crossover_cpu(population, k)
    
    new_children = []
    seen = set()
    
    # Process pairs in batches on GPU
    num_pairs = len(population) // 2
    
    for pair_idx in range(num_pairs):
        i = pair_idx * 2
        if i + 1 >= len(population):
            break
            
        parent1 = cp.asarray(population[i]["bit_string"], dtype=cp.int32)
        parent2 = cp.asarray(population[i + 1]["bit_string"], dtype=cp.int32)
        
        # Generate crossover points
        chromo_len = len(parent1)
        points = sorted(random.sample(range(1, chromo_len - 1), min(k, chromo_len - 2)))
        points = [0] + points + [chromo_len]
        
        # Create masks for crossover segments
        child1_gpu = cp.zeros_like(parent1)
        child2_gpu = cp.zeros_like(parent2)
        
        for seg_idx in range(len(points) - 1):
            start = points[seg_idx]
            end = points[seg_idx + 1]
            
            if seg_idx % 2 == 0:
                child1_gpu[start:end] = parent1[start:end]
                child2_gpu[start:end] = parent2[start:end]
            else:
                child1_gpu[start:end] = parent2[start:end]
                child2_gpu[start:end] = parent1[start:end]
        
        # Convert back to CPU and validate
        child1 = cp.asnumpy(child1_gpu).tolist()
        child2 = cp.asnumpy(child2_gpu).tolist()
        
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

def bit_flip_mutator(population: Population, k: int, use_gpu=None):
    """Bit flip mutation with optional GPU acceleration."""
    if use_gpu is None:
        use_gpu = USE_GPU and GPU_AVAILABLE and CUPY_AVAILABLE
    
    if use_gpu and k > 5:
        return bit_flip_mutator_gpu(population, k)
    else:
        return bit_flip_mutator_cpu(population, k)


def bit_flip_mutator_cpu(population: Population, k: int):
    """CPU-based bit flip mutation (original implementation)."""
    new_chromos = []
    for i in range(k):
        new_chromos.append(bit_flip_mutation(population[i]["bit_string"]))
    return new_chromos


def bit_flip_mutator_gpu(population: Population, k: int):
    """GPU-accelerated batch bit flip mutation."""
    if cp is None:
        # Fallback to CPU if CuPy is not available
        return bit_flip_mutator_cpu(population, k)
    
    new_chromos = []
    
    # Process in batches on GPU
    batch_size = min(32, k)
    for i in range(0, k, batch_size):
        batch_end = min(i + batch_size, k)
        batch_indices = list(range(i, batch_end))
        
        # Convert batch to GPU array
        batch_chromos = [population[idx]["bit_string"] for idx in batch_indices]
        batch_array = cp.asarray(batch_chromos, dtype=cp.int32)
        
        # Generate random flip positions for each chromosome
        chromo_len = len(batch_chromos[0])
        flip_positions = cp.random.randint(0, chromo_len, size=(len(batch_indices),))
        
        # Perform bit flips using GPU array operations
        for j, pos in enumerate(flip_positions):
            batch_array[j, pos] = 1 - batch_array[j, pos]
        
        # Convert back to CPU
        mutated_batch = cp.asnumpy(batch_array).tolist()
        new_chromos.extend(mutated_batch)
    
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

def complement_mutator(population: Population, k: int, use_gpu=None):
    """Complement mutation with optional GPU acceleration."""
    if use_gpu is None:
        use_gpu = USE_GPU and GPU_AVAILABLE and CUPY_AVAILABLE
    
    if use_gpu and k > 5:
        return complement_mutator_gpu(population, k)
    else:
        return complement_mutator_cpu(population, k)


def complement_mutator_cpu(population: Population, k: int):
    """CPU-based complement mutation (original implementation)."""
    new_chromos = []
    for i in range(k):
        new_chromos.append(Complement_mutation(population[i]["bit_string"]))
    return new_chromos


def complement_mutator_gpu(population: Population, k: int):
    """GPU-accelerated batch complement mutation."""
    if cp is None:
        # Fallback to CPU if CuPy is not available
        return complement_mutator_cpu(population, k)
    
    new_chromos = []
    
    # Process in batches on GPU
    batch_size = min(32, k)
    for i in range(0, k, batch_size):
        batch_end = min(i + batch_size, k)
        batch_chromos = [population[idx]["bit_string"] for idx in range(i, batch_end)]
        
        # Convert to GPU array and complement (1 - array)
        batch_array = cp.asarray(batch_chromos, dtype=cp.int32)
        batch_array = 1 - batch_array  # Vectorized complement operation
        
        # Convert back to CPU
        mutated_batch = cp.asnumpy(batch_array).tolist()
        new_chromos.extend(mutated_batch)
    
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

def reverse_mutator(population: Population, k: int, use_gpu=None):
    """Reverse mutation with optional GPU acceleration."""
    if use_gpu is None:
        use_gpu = USE_GPU and GPU_AVAILABLE and CUPY_AVAILABLE
    
    if use_gpu and k > 5:
        return reverse_mutator_gpu(population, k)
    else:
        return reverse_mutator_cpu(population, k)


def reverse_mutator_cpu(population: Population, k: int):
    """CPU-based reverse mutation (original implementation)."""
    new_chromos = []
    for i in range(k):
        new_chromos.append(reverse_mutation(population[i]["bit_string"]))
    return new_chromos


def reverse_mutator_gpu(population: Population, k: int):
    """GPU-accelerated batch reverse mutation."""
    if cp is None:
        # Fallback to CPU if CuPy is not available
        return reverse_mutator_cpu(population, k)
    
    new_chromos = []
    
    # Process in batches on GPU
    batch_size = min(32, k)
    for i in range(0, k, batch_size):
        batch_end = min(i + batch_size, k)
        batch_chromos = [population[idx]["bit_string"] for idx in range(i, batch_end)]
        
        # Convert to GPU array and reverse using flip
        batch_array = cp.asarray(batch_chromos, dtype=cp.int32)
        batch_array = cp.flip(batch_array, axis=1)  # Reverse along chromosome axis
        
        # Convert back to CPU
        mutated_batch = cp.asnumpy(batch_array).tolist()
        new_chromos.extend(mutated_batch)
    
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

def rotation_mutator(population: Population, k: int, use_gpu=None):
    """Rotation mutation with optional GPU acceleration."""
    if use_gpu is None:
        use_gpu = USE_GPU and GPU_AVAILABLE and CUPY_AVAILABLE
    
    if use_gpu and k > 5:
        return rotation_mutator_gpu(population, k)
    else:
        return rotation_mutator_cpu(population, k)


def rotation_mutator_cpu(population: Population, k: int):
    """CPU-based rotation mutation (original implementation)."""
    new_chromos = []
    for i in range(k):
        new_chromos.append(Rotation_mutation(population[i]["bit_string"]))
    return new_chromos


def rotation_mutator_gpu(population: Population, k: int):
    """GPU-accelerated batch rotation mutation."""
    if cp is None:
        # Fallback to CPU if CuPy is not available
        return rotation_mutator_cpu(population, k)
    
    new_chromos = []
    
    # Process in batches on GPU
    batch_size = min(32, k)
    for i in range(0, k, batch_size):
        batch_end = min(i + batch_size, k)
        batch_chromos = [population[idx]["bit_string"] for idx in range(i, batch_end)]
        
        # Convert to GPU array
        batch_array = cp.asarray(batch_chromos, dtype=cp.int32)
        chromo_len = len(batch_chromos[0])
        
        # Generate random rotation amounts for each chromosome
        rotations = cp.random.randint(1, chromo_len, size=(len(batch_chromos),))
        
        # Apply rotations using GPU array operations
        for j, rot in enumerate(rotations):
            batch_array[j] = cp.roll(batch_array[j], int(rot))
        
        # Convert back to CPU
        mutated_batch = cp.asnumpy(batch_array).tolist()
        new_chromos.extend(mutated_batch)
    
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

def merge_GAs(all_results: List[List[Generation]], number_of_runs: int, number_of_generations: int) -> Merged_GA:
    merged_results: Merged_GA = []

    # Iterate over each generation index
    for gen_idx in range(number_of_generations):
        # Collect the same generation across all runs
        gen_across_runs = [all_results[run_idx][gen_idx] for run_idx in range(number_of_runs)]

        # Aggregate statistics
        total_gen_fitness = sum(g["total_fitness"] for g in gen_across_runs)
        average_gen_fitness = total_gen_fitness / number_of_runs

        # Find best and worst chromosomes across runs
        best_chromo = max((g["best_chromosome"] for g in gen_across_runs), key=lambda c: c["fitness"])
        worst_chromo = min((g["worst_chromosome"] for g in gen_across_runs), key=lambda c: c["fitness"])

        gen_size = gen_across_runs[0]["gen_size"]

        # Construct merged generation
        merged_gen: Merged_Generation = {
            "best_chromosome": best_chromo,
            "worst_chromosome": worst_chromo,
            "gen_size": gen_size,
            "total_generations_fitness": total_gen_fitness,
            "average_generations_fitness": average_gen_fitness
        }

        merged_results.append(merged_gen)

    return merged_results

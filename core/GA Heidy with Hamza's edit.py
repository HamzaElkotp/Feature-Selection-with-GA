# Heidy Salem
# 20230641
# GA for feature selection
from re import S
import numpy as np
import array

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


# -> a fun chromosome return array of random binary bits (0,1)
# create an initial random array (chromosome) of size num
# num_features is the number of features in the dataset
# size is the size of the chromosome (the number of bits in the chromosome)
# the function gets num_features as an input and returns a random array of binary bits (0,1)
# Returns tuple instead of np object to be able to pass the chromosome to the validate function

def create_bitstring_chromosome(num_features: int) -> tuple:
    return tuple(np.random.randint(2, size=num_features)) #randint is a function that generates random integers between 0 and 1 from numpy library


# validate the chromosome based on some criteria, and then return true if valid, and false if invalid chromosome
def validate_bitstring_chromosome(chromosome:tuple) -> bool :
    selected_indices = chromosome.count(1)
    if selected_indices == 0: # check if chromosome has no ones eg: [0 0 0 0]
        return False
    return True


# -> a fun population  return array of random chromosomes 

# First : initiale (random) population:
# population_size is the number of chromosomes in the population
# num_features is the number of features in the dataset
# the function gets population_size and num_features as an input and returns an array of random chromosomes (each chromosome is an array of binary bits (0,1))
# the function uses the create_bitstring_chromosome function to create each chromosome
# the function uses a list comprehension to create a list of random chromosomes
# the function uses the create_bitstring_chromosome function to create each chromosome
# the function returns a list of random chromosomes
# returns an array of valid unique chromosomes

def initialize_population(population_size: int , num_features: int):
    if (population_size < 2):
        raise ValueError("population_size must be at least 2")
    if (population_size >= pow(2, num_features)):
        raise ValueError("population_size must be less than 2^num_features.")
    if (num_features < 2):
        raise ValueError("num_features must be at least 2")

    population = set()
    while len(population) < population_size: # generate chromosomes until generate the full population with only unique valid chromosomes
        chrom = create_bitstring_chromosome(num_features)
        if validate_bitstring_chromosome(chrom): # validate chromosome
            population.add(chrom)
    return [np.array(c) for c in population] # the function uses a list comprehension to create a list of random chromosomes


# the function returns a list of random chromosomes (n number of chromosomes = 20 which is the population size and each chromosome is an array of binary bits (0,1) of size 12 which is the number of features in the dataset)
# the function uses a list comprehension to create a list of random chromosomes

population = initialize_population(population_size=20, num_features=12)
for chromo in population:
    print(chromo)




from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# the function uses the np.where function to get the indices of the selected features
# the function uses the cross_val_score function to get the accuracy of the model
# This function computes the accuracy of the model after trained with the selected features
def compute_accuracy(chromosome, dataset_features, prediction_target) -> np.floating:
    selected_indices = np.where(chromosome == 1)[0]  # the function uses the np.where function to get the indices of the selected features

    #  Use .iloc for integer-location based indexing with pandas DataFrames
    dataset_features_selected = dataset_features.iloc[:, selected_indices]  # the function uses the np.where function to get the indices of the selected features

    model = DecisionTreeClassifier()  # the function uses the DecisionTreeClassifier function to create a decision tree model

    # Accuracy using cross-validation
    accuracy = np.mean(cross_val_score(model, dataset_features_selected, prediction_target, cv=3))
    # the function uses the cross_val_score function to get the accuracy of the model

    return accuracy


#-> a fun fitness return one chromosome fitness 
# the function gets chromosome, Dataset_features, prediction_target, Beta as an input and returns the fitness of the chromosome
# the function uses the np.where function to get the indices of the selected features
# the function uses the len function to get the number of selected features
# the function returns the fitness of the chromosome
# fitness formula:
# fitness=accuracy*alpha−Beta⋅( Features_selected / Features_total  ) 
#Fitness Function for One Chromosome

def compute_fitness(chromosome, dataset_features, prediction_target, alpha=1, beta=1) -> float:
    """
    Compute fitness for one chromosome.
    chromosome: 0/1 numpy array
    Dataset_features: dataset features
    prediction_target: prediction target
    penalty: penalty for the fitness = Beta * (Features_selected / Features_total)
    """

    selected_indices = np.where(chromosome == 1)[0]
    accuracy = compute_accuracy(chromosome, dataset_features, prediction_target)

    # Penalty
    features_total = len(chromosome)  # = len(Dataset_features)
    features_selected = len(selected_indices)
    penalty = beta * (features_selected / features_total)

    # Reward
    reward = alpha * accuracy

    fitness_value:float = float(reward - penalty)
    return fitness_value


# Function to Calculate Fitness for Whole Population
# Returns a List of chromosome and value of each fitness for this chromosome
#-> a function  evaluate population take the array population and call compute fitness for each chromosome in the population and return a dictionary of key index chromosome and value of each fitness for this chromosome   
# the function gets population, Dataset_features, prediction_target, Beta as an input and returns a dictionary of chromosome and value of each fitness for this chromosome
# the function uses a for loop to iterate over the population
# the function uses the compute_fitness function to get the fitness of each chromosome
# the function returns a dictionary of chromosome and value of each fitness for this chromosome
# Function to Calculate Fitness for Whole Population

def get_population_fitness(_population, dataset_features, prediction_target, alpha=1, beta=1):
    """
        Returns a list of fitness values for each chromosome in the population.
    """
    population_with_fitness:Population = []

    for chrom in _population:
        fitness = compute_fitness(chrom, Dataset_features, prediction_target, alpha, beta)
        chrom = Chromosome(bit_string=chrom, fitness=fitness)
        population_with_fitness.append(chrom)

    return population_with_fitness




import pandas as pd
# Importing the dataset
df = pd.read_csv('healthcare_dataset.csv')

# Define features and target
prediction_target = df['Test Results']

# Drop columns that are unlikely to be useful as features or are high cardinality strings/dates
Dataset_features = df.drop(columns=['Test Results', 'Name', 'Doctor', 'Hospital', 'Insurance Provider', 'Date of Admission', 'Discharge Date'])

# Identify categorical columns for one-hot encoding
categorical_cols = Dataset_features.select_dtypes(include='object').columns

# Apply one-hot encoding to categorical columns
Dataset_features = pd.get_dummies(Dataset_features, columns=categorical_cols, drop_first=True)

population = initialize_population(population_size=20, num_features=len(Dataset_features.columns))

fitness_list = get_population_fitness(population, Dataset_features, prediction_target)

print(fitness_list)



# Genetic Algorithm Selection methods 
# first random parents Selection method (the easiest method for selection)

# Version 1 — Select 2 random parents
# the function gets population as an input and returns 2 random parents
# the function uses the random.choice function to select 2 random parents
# the function returns 2 random parents (parent1 and parent2)

import random

def random_selection(population):
    parent1 = random.choice(population)#fun random.choice -> 
    parent2 = random.choice(population)
    return parent1, parent2

p1, p2 = random_selection(population)
print(p1)
print(p2)

def random_selection_2(population, k=2):
    return random.sample(population, 2)# -> Note:
 # random.sample chooses unique parents (no repetition).

p1, p2 = random_selection_2(population)
print(p1)
print(p2)

    #----------------------------------------------#
    #Version 2 — Select k parents
# the function gets population and k as an input and returns k random parents
# the function uses the random.sample function to select k random parents
# the function returns k random parents

def random_selection_k(population, k):
    return random.sample(population, k)# -> Note:
# random.sample chooses unique parents (no repetition).

p1, p2, p3, p4 = random_selection_k(population,4)
print(p1)
print(p2)
print(p3)
print(p4)

#----------------------------------------------#

# Second. Roulette Wheel Selection (Fitness Proportionate)
#Idea :
# Chromosomes with higher fitness get bigger slice of the wheel.
#Selection probability = fitness / total_fitness.

####################   ⚠ Important:    ####################
# Fitness must be positive. If not, you must shift values.
#
# Shift Fitness Values (to be positive)


# Shift fitness to ensure all values are positive
def shift_fitness(fitness_list):
    # Extract fitness values from dict
    fitnesses = [fitness_list[i]["fitness"] for i in fitness_list]

    min_f = min(fitness_list)

    # Shift values only if needed
    if min_f <= 0:
        shift_amount = abs(min_f) + 1e-6
        fitnesses = [f + shift_amount for f in fitness_list]

    return fitnesses

fitnesses = shift_fitness(fitness_list)
# Now we can do : Roulette Wheel Selection (Fitness Proportionate) Safetly


# Function: Descending_order_fitnesses
def Descending_order_fitnesses(population_with_fitness:Population) -> Population:
    return sorted(
        population_with_fitness,
        key=lambda chrom: chrom["fitness"],
        reverse=True
    )

# Function: Descending_order_ratios :
def Descending_order_ratios(sorted_population:Population):
    total_fitness = sum(chrom["fitness"] for chrom in sorted_population) # before it was `total_fitness = sum(fitnesses)`

    ratios = [(chromo["fitness"] / total_fitness) * 100 for chromo in sorted_population]
    
    return ratios #[50,30,20]


def roulette_wheel(ratios):
    """
    Given a list of ratios (probabilities),
    return the cumulative roulette wheel list.
    Example:
        [50, 30, 20] -> [50, 80, 100]
    """

    roulette = []
    cumulative_sum = 0

    for r in ratios:
        cumulative_sum += r
        roulette.append(cumulative_sum)

    return roulette  #->-> [50, 80, 100]


ratios = Descending_order_ratios(list1, fitnesses)
roulette_list = roulette_wheel(ratios)

print(roulette_list)

import random

def roulette_wheel_selection(ratio_list, list2, population):
    # random number between 0 and 100
    r = random.uniform(0, ratio_list[-1]) #->70

    # Find the first interval where r <= roulette_list[i]
    for i in range(len(ratio_list)):
        if r <= ratio_list[i] :
            return population[i]  # return the chromosome
        

selected = roulette_wheel_selection(roulette_list, population)
print(selected)

################################################################################################################### 
###################################################################################################################
###################################################################################################################
###################################################################################################################

############## Cross over methods #############
## First : single point crossover

def single_point_crossover(p1, p2):
    """
    Perform single-point crossover between two parent chromosomes.
    Assumes both parents are sequences of the same length.
    """
    chromo_len = len(p1)
    # Choose a crossover point between 1 and chromo_len-1 (inclusive)
    r = random.sample(1, chromo_len - 1)
    offspring1 = p1[:r] + p2[r:]
    offspring2 = p2[:r] + p1[r:]
    return offspring1, offspring2

##  Second : Two point crossover

def Two_point_crossover(p1, p2):
    """
    Perform single-point crossover between two parent chromosomes.
    Assumes both parents are sequences of the same length.
    """
    chromo_len = len(p1)
    # Choose a crossover point between 1 and chromo_len-1 (inclusive)
    r1 = random.sample(range(1, chromo_len - 1), k=1)
    r2 = random.sample(range(1, chromo_len - 1), k=1)
    if (r1>r2):
        offspring1 = p1[:r2] + p2[r2:r1] + p1[r1:]
        offspring2 = p2[:r2] + p1[r2:r1] + p2[r1:]
    else:
        offspring1 = p1[:r1] + p2[r1:r2] + p1[r2:]
        offspring2 = p2[:r1] + p1[r1:r2] + p2[r2:]

    return offspring1, offspring2



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
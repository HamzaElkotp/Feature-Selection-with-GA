# Heidy Salem
# 20230641
# GA for feature selection
from re import S
import numpy as np
import array

# -> a fun chromosome return array of random binary bits (0,1)
# create an initial random array (chromosome) of size num
# num_features is the number of features in the dataset
# size is the size of the chromosome (the number of bits in the chromosome)
# the function gets num_features as an input and returns a random array of binary bits (0,1)

def create_bitstring_chromosome(num_features: int) :
    chromosome = np.random.randint(2, size=num_features)#randint is a function that generates random integers between 0 and 1 from numpy library
    return chromosome

# -> a fun population  return array of random chromosomes 

# First : initiale (random) population:
# population_size is the number of chromosomes in the population
# num_features is the number of features in the dataset
# the function gets population_size and num_features as an input and returns an array of random chromosomes (each chromosome is an array of binary bits (0,1))
# the function uses the create_bitstring_chromosome function to create each chromosome
# the function uses a list comprehension to create a list of random chromosomes
# the function uses the create_bitstring_chromosome function to create each chromosome
# the function returns a list of random chromosomes
 

def initialize_population(population_size: int , num_features: int):
    return [create_bitstring_chromosome(num_features) for _ in range(population_size)]# the function uses a list comprehension to create a list of random chromosomes

# the function returns a list of random chromosomes (n number of chromosomes = 20 which is the population size and each chromosome is an array of binary bits (0,1) of size 12 which is the number of features in the dataset)
# the function uses a list comprehension to create a list of random chromosomes

population = initialize_population(population_size=20, num_features=12)
for chromo in population:
    print(chromo)




from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#-> a fun fitness return one chromosome fitness 
# the function gets chromosome, Dataset_features, prediction_target, Beta as an input and returns the fitness of the chromosome
# the function uses the np.where function to get the indices of the selected features
# the function uses the len function to get the number of selected features
# the function uses the cross_val_score function to get the accuracy of the model
# the function returns the fitness of the chromosome
# fitness formula:
# fitness=accuracy*alpha−Beta⋅( Features_selected / Features_total  ) 
#Fitness Function for One Chromosome

def compute_fitness(chromosome, Dataset_features, prediction_target, alpha=1,  Beta=1):
    """
    Compute fitness for one chromosome.
    chromosome: 0/1 numpy array
    Dataset_features: dataset features
    prediction_target: prediction target
    penalty: penalty for the fitness = Beta * (Features_selected / Features_total)
    """
    selected_indices = np.where(chromosome == 1)[0]# the function uses the np.where function to get the indices of the selected features


    # check if no feature selected 
    if len(selected_indices) == 0:# the function uses the len function to get the number of selected features
        return 0.0# if no feature selected, the fitness is 0
     
     #  Use .iloc for integer-location based indexing with pandas DataFrames
    Dataset_features_selected = Dataset_features.iloc[:, selected_indices]# the function uses the np.where function to get the indices of the selected features
    # Model
    model = DecisionTreeClassifier()# the function uses the DecisionTreeClassifier function to create a decision tree model

    # Accuracy using cross-validation
    # the function uses the cross_val_score function to get the accuracy of the model
    accuracy = np.mean(cross_val_score(model, Dataset_features_selected, prediction_target, cv=3))

    # Penalty
    Features_total = len(chromosome) #  = len(Dataset_features)
    Features_selected = len(selected_indices)
    penalty = Beta * (Features_selected / Features_total)

    fitness_value = (alpha * accuracy) - penalty
    return fitness_value

# Function to Calculate Fitness for Whole Population
# Returns a List of chromosome and value of each fitness for this chromosome
#-> a function  evaluate population take the array population and call compute fitness for each chromosome in the population and return a dictionary of key index chromosome and value of each fitness for this chromosome   
# the function gets population, Dataset_features, prediction_target, Beta as an input and returns a dictionary of chromosome and value of each fitness for this chromosome
# the function uses a for loop to iterate over the population
# the function uses the compute_fitness function to get the fitness of each chromosome
# the function returns a dictionary of chromosome and value of each fitness for this chromosome
# Function to Calculate Fitness for Whole Population

def get_fitness_list(population, Dataset_features, prediction_target):
    """
    Returns a list of fitness values for each chromosome in the population.
    """
    fitness_list = []

    for chromosome in population:
        fitness = compute_fitness(chromosome, Dataset_features, prediction_target, alpha=1, Beta=1)
        fitness_list.append(fitness)

    return fitness_list

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

fitness_list = get_fitness_list(population, Dataset_features, prediction_target)

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
def Descending_order_fitnesses(fitnesses):
    """
    Returns:
        list1: fitness values sorted in descending order
        list2: indices of chromosomes arranged according to sorted fitness
    """

    # Pair index with fitness
    indexed_fitness = list(enumerate(fitnesses))  #fitnesses =[30,20,40]
    #Ex:
    # indexed_fitness = [(0,30), (1,20), (2,40)]

    # Sort by fitness DESC (key = fitness)
    sorted_pairs = sorted(indexed_fitness, key=lambda x: x[1], reverse=True)
    #Ex:
    # sorted_pairs = [(2,40), (0,30), (1,20)]

    # des_order_fitnesses: sorted fitness values
    des_order_fitnesses = [pair[1] for pair in sorted_pairs]
    #des_order_fitnesses = [40,30,20]

    # des_order_indices: original indices arranged based on sorted fitness
    des_order_indices = [pair[0] for pair in sorted_pairs]
    # des_order_indices: [2,0,1]

    return des_order_fitnesses, des_order_indices

list1, list2 = Descending_order_fitnesses(fitnesses)

print("Descending Fitness:", list1)
print("Sorted Indexes:", list2)

# Function: Descending_order_ratios :
def Descending_order_ratios(list1, fitnesses): #list1 =[0.5,0.3,0.2] 
    total_fitness = sum(fitnesses)#1
    ratios =[]
    for fit in list1:
        ratio =(fit / total_fitness)*100 # -> 0.5/1*100=50
        ratios.append(ratio) 
    
    return ratios #[50,30,20]

des_r = Descending_order_ratios(list1, fitnesses)

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

def roulette_wheel_selection(roulette_list, list2, population):
    # random number between 0 and 100
    r = random.uniform(0, 100) #->70

    # Find the first interval where r <= roulette_list[i]
    for i in range(len(roulette_list)): 
        if r <= roulette_list[i] :    
            selected_index = list2[i]          # original index in population
            return population[selected_index]  # return the chromosome
        

selected = roulette_wheel_selection(roulette_list, list2, population)
print(selected)

################################################################################################################### 
###################################################################################################################
###################################################################################################################
###################################################################################################################

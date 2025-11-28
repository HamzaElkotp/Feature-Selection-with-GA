# Heidy Salem
# 20230641
# GA for feature selection
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
# fitness=accuracy−Beta⋅( Features_selected / Features_total  ) 
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
    Features_total = len(chromosome)
    Features_selected = len(selected_indices)
    penalty = Beta * (Features_selected / Features_total)

    fitness_value = (alpha * accuracy) - penalty
    return fitness_value

# Function to Calculate Fitness for Whole Population
# Returns a dictionary of chromosome and value of each fitness for this chromosome
#-> a function  evaluate population take the array population and call compute fitness for each chromosome in the population and return a dictionary of key index chromosome and value of each fitness for this chromosome   
# the function gets population, Dataset_features, prediction_target, Beta as an input and returns a dictionary of chromosome and value of each fitness for this chromosome
# the function uses a for loop to iterate over the population
# the function uses the compute_fitness function to get the fitness of each chromosome
# the function returns a dictionary of chromosome and value of each fitness for this chromosome
# Function to Calculate Fitness for Whole Population

def evaluate_population(population, Dataset_features, prediction_target, Beta=1):
    fitness_dict = {}

    for index, chromosome in enumerate(population):
        fitness = compute_fitness(chromosome, Dataset_features, prediction_target, Beta)
        fitness_dict[index] = {
            "chromosome": chromosome,
            "fitness": fitness
            }
        # the function uses a dictionary to store the chromosome and the fitness of the chromosome
    return fitness_dict
# the function returns a dictionary of chromosome and value of each fitness for this chromosome

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
fitness_results = evaluate_population(population, Dataset_features, prediction_target)

print(fitness_results)
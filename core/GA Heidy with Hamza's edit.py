# Heidy Salem
# 20230641
# GA for feature selection
import numpy as np
import array




# Chromosome class
# Generation Class
# Population Class
# GA Interface









# -> a fun chromosome return array of random binary bits ( 0,1)
# create an initial random array (chromosome) of size num

def create_bitstring_chromosome(num_features: int) :
    chromosome = np.random.randint(2, size=num_features)
    return chromosome

# > a fun population  return array of random chromosomes 

# First : initiale (random) population:
def initialize_population(population_size: int , num_features: int):
    return [create_bitstring_chromosome(num_features) for _ in range(population_size)]

 # First : initiale (random) population:
population = initialize_population(population_size=20, num_features=12)
for chromo in population:
    print(chromo)

#Fitness Function for One Chromosome

#-> a fun fitness return one chromosome fitness 

# fitness formula:
# fitness=α⋅accuracy−α⋅( n selected / m ) 

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def compute_fitness(chromosome, X, y, alpha=1, beta=1):
    """
    Compute fitness for one chromosome.
    chromosome: 0/1 numpy array
    X: dataset features
    y: labels
    """
    selected_indices = np.where(chromosome == 1)[0]

    # If no feature selected → very low fitness
    if len(selected_indices) == 0:
        return 0.0

    X_selected = X[:, selected_indices]

    # Model
    model = DecisionTreeClassifier()

    # Accuracy using cross-validation
    accuracy = np.mean(cross_val_score(model, X_selected, y, cv=3))

    # Penalty
    m = len(chromosome)
    n_selected = len(selected_indices)
    penalty = beta * (n_selected / m)

    fitness_value = (alpha * accuracy) - penalty
    return fitness_value

# Function to Calculate Fitness for Whole Population
# Returns a dictionary of key index chromosome and value of each fitness for this chromosome
#-> a function  evaluate population take the array population and call compute fitness for each chromosome in the population and return a dictionary of key index chromosome and value of each fitness for this chromosome   

def evaluate_population(population, X, y, alpha=1, beta=1):
    fitness_dict = {}

    for index, chromosome in enumerate(population):
        fitness = compute_fitness(chromosome, X, y, beta, alpha)
        fitness_dict[index] = fitness

    return fitness_dict

    population = initialize_population(population_size=20, num_features=12)

X = df.drop("Target", axis=1).values # drop the target column and get the values (all the features)
y = df["Target"].values # target is the column name of the target variable (the label) (the out come result)

fitness_results = evaluate_population(population, X, y)

print(fitness_results)


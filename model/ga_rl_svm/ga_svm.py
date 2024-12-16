from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from functools import cmp_to_key
import copy
from ga import *
from data import *
from sklearn import svm
import svm_rl

# Parameters for the genetic algorithm
pop_size = 500  # Population size
max_gen = 20    # Maximum number of generations

# Define an individual in the genetic algorithm
class SVMIndividual:
    def __init__(self, bin_len):
        # Generate a random binary code representing selected features
        self.bin_code = gen(bin_len)  
        self.fitness = 0
        self.status = [i for i, val in enumerate(self.bin_code) if val == '1']

    def update_status(self):
        """Update the selected features based on the binary code."""
        self.status = [i for i, val in enumerate(self.bin_code) if val == '1']

    def output(self):
        """Print the individual's binary code, selected features, and fitness."""
        print('Binary Code:', self.bin_code)
        print('Selected Features:', self.status)
        print('Fitness:', self.fitness)

# Class for tracking the optimization process
class GeneticAlgorithmTracker:
    def __init__(self):
        self.pop = []  # Population at each generation
        self.acc = []  # Accuracy at each generation
        self.smi = []  # Similarity metrics (if applicable)
        self.fit = []  # Fitness scores
        self.times = 0  # Number of iterations

# Extract features based on the selected binary code
def get_data_subset(data, res, individual):
    """Filter data based on the features selected by the individual."""
    individual.update_status()
    filtered_data = data[individual.status]
    filtered_val_data = val_data[individual.status]
    return filtered_data, filtered_val_data

# Evaluate the fitness of an individual using SVM
def evaluate_fitness(data, res, individual, debug=False):
    """Calculate the fitness of an individual using SVM classification accuracy."""
    individual.update_status()
    filtered_data, filtered_val_data = get_data_subset(data, res, individual)

    # Train an SVM classifier
    clf = svm.LinearSVC(C=1, loss='hinge')
    clf.fit(filtered_data, res)

    # Evaluate the model on validation data
    score = clf.score(filtered_val_data, val_res)

    if debug:
        print(f"Fitness Score: {score}")
    return score

# Main genetic algorithm
def genetic_algorithm(population, max_gen=30):
    tracker = GeneticAlgorithmTracker()
    best_1 = copy.deepcopy(sort_indi(population)[0])
    best_2 = copy.deepcopy(sort_indi(population)[1])

    while max_gen > 0:
        print(f"Generation {30 - max_gen} in progress...")
        next_gen = [copy.deepcopy(best_1), copy.deepcopy(best_2)]

        # Generate new individuals through crossover and mutation
        while len(next_gen) < len(population):
            parent_1, parent_2 = random_select(population)
            child_1, child_2 = crossover(parent_1, parent_2, rate=0.7)
            child_1 = mutate(child_1)
            child_2 = mutate(child_2)

            # Evaluate the fitness of new individuals
            child_1.fitness = evaluate_fitness(train_data, train_res, child_1)
            child_2.fitness = evaluate_fitness(train_data, train_res, child_2)

            next_gen.extend([child_1, child_2])

        # Keep the top individuals from the current and previous generations
        population = list(filter(None.__ne__, next_gen))
        population = sort_indi(population)[:len(population)]
        best_1 = copy.deepcopy(sort_indi(population)[0])
        best_2 = copy.deepcopy(sort_indi(population)[1])

        # Record metrics for tracking
        tracker.pop.append(copy.deepcopy(best_1))
        tracker.acc.append(best_1.fitness)

        max_gen -= 1
        print(f"Generation {30 - max_gen} completed.\n")

    return tracker

# Initialize the population
population = []
for _ in range(pop_size):
    individual = SVMIndividual(41)  # 41 features in total
    individual.fitness = evaluate_fitness(train_data, train_res, individual)
    population.append(individual)

# Run the genetic algorithm
results = genetic_algorithm(population, max_gen)

# Analyze the results to determine the most frequently selected features
feature_count = [0] * 41
for individual in results.pop:
    for feature in individual.status:
        feature_count[feature] += 1

# Select features that appear in more than half of the generations
final_features = [i for i, count in enumerate(feature_count) if count > max_gen / 2]
print("Final Selected Features:", final_features)
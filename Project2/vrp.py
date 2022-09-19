import csv
import math
import random
from dataclasses import dataclass, field
import time
import sys

import matplotlib.pyplot as plt

from typing import List

NO_GENERATIONS = 3000
POPULATION_SIZE = 50
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.05
OVER_WEIGHT_PENALTY = 1000
SELECTION_PRESSURE = 1.5
KEEP_BEST = True

NO_VEHICLES = 9


# TODO crossover anpassen, check dass distance matrix korrekt ist, ansonsten infeasible solution rauswerfen


@dataclass
class Chromosome:
    stops: List[int] = field(default_factory=list)
    vehicles: List[int] = field(default_factory=list)
    fitness: int = 0


def gen_chromosome():
    # generates a random sequence of customers
    # and a corresponding array which vehicle stops at this customer
    _stops = list(range(1, 55))
    random.shuffle(_stops)
    _vehicles = []
    for i in range(0, len(_stops)):
        _vehicles.append(random.randint(0, 8))

    return Chromosome(_stops.copy(), _vehicles.copy())


def gen_population():
    _chromosomes = []
    for i in range(0, POPULATION_SIZE):
        _chromosomes.append(gen_chromosome())
    return _chromosomes


# applying the fitness function using a penalty if the weight of a vehicle is more than 100
def fun_fitness(costs, weight):
    if weight > 100:
        weight_penalty = (weight - 100) * OVER_WEIGHT_PENALTY
    else:
        weight_penalty = 0
    fitness = costs + weight_penalty
    return fitness


# calculate the path_costs and vehicle_weights
def calculate_path_costs_and_weights(c: Chromosome):
    path_costs = [0] * 9
    vehicle_weights = [0] * 9
    prev_stop = [0] * 9

    for i in range(0, len(c.vehicles)):
        stop = c.stops[i]  # the current stop
        vehicle_no = c.vehicles[i]  # the current driver that stops for this customer
        dist = distance_matrix[prev_stop[vehicle_no]][stop]  # distance driver makes for this customer
        path_costs[vehicle_no] += dist
        vehicle_weights[vehicle_no] += demands[stop]
        prev_stop[vehicle_no] = stop

    # calculate costs for return to depot
    for i in range(0, len(prev_stop)):
        return_dist = distance_matrix[prev_stop[i]][0]
        path_costs[i] += return_dist

    return path_costs, vehicle_weights


# calculates the fitness of a chromosome, the higher the fitness the better is the chromosome
def evaluate_fitness(c: Chromosome):
    path_costs, vehicle_weights = calculate_path_costs_and_weights(c)

    total_fitness = 0
    for i in range(0, len(path_costs)):
        f = fun_fitness(path_costs[i], vehicle_weights[i])
        total_fitness += f

    c.fitness = 1 / total_fitness


def scale_fitness(r, N):
    return 2 - SELECTION_PRESSURE + 2 * (SELECTION_PRESSURE - 1) * ((r - 1) / (N - 1))


# select the parent using the rank selection
def select_parent_rank_selection(chromosomes):
    total_fitness = 0
    chroms_fitness = []
    for chrom in chromosomes:
        total_fitness += chrom.fitness
        chroms_fitness.append(chrom.fitness)

    # gives the ranks of the chromosomes according to their fitness where a higher rank is better
    # note that the rank goes from 0 to N-1, for the formula we need 1 up to N, hence add +1
    ranks = [sorted(chroms_fitness).index(x) + 1 for x in chroms_fitness]
    chroms_fitness_scaled = [scale_fitness(r, len(ranks)) for r in ranks]
    total_fitness_scaled = sum(chroms_fitness_scaled)

    if total_fitness_scaled == 0:
        chroms_fitness_scaled[0] += 1

    # create the selection probabilities from the scaled fitness
    selection_probabilities = [f_s / total_fitness_scaled for f_s in chroms_fitness_scaled]

    selected_chrom = random.choices(chromosomes, weights=selection_probabilities)[0]

    return selected_chrom


# select the parent using the roulette wheel selection
def select_parent_roulette_selection(chromosomes):
    total_fitness = 0
    chroms_fitness = []
    for chrom in chromosomes:
        total_fitness += chrom.fitness
        chroms_fitness.append(chrom.fitness)

    # create the selection probabilities from the scaled fitness
    selection_probabilities = [f_s / total_fitness for f_s in chroms_fitness]

    selected_chrom = random.choices(chromosomes, weights=selection_probabilities)[0]

    return selected_chrom


# TODO can most probably be deleted just kept in case we need it
# do the crossover, implemented according to the order crossover
def do_crossover_old(parent1: Chromosome, parent2: Chromosome):
    crossover_point_1 = random.randint(0, len(parent1.stops) - 1)
    crossover_point_2 = random.randint(0, len(parent1.stops) - 1)

    child_stops = [-1] * len(parent1.stops)
    child_vehicles = [-1] * len(parent1.vehicles)
    used_values = []

    for i in range(min(crossover_point_1, crossover_point_2), max(crossover_point_1, crossover_point_2) + 1):
        child_stops[i] = parent1.stops[i]
        used_values.append(parent1.stops[i])
        child_vehicles[i] = parent1.vehicles[i]

    available_values = [ele for ele in parent2.stops if ele not in used_values]

    for i in range(0, len(parent1.stops)):
        if child_stops[i] == -1:
            index_of_no_in_parent2 = parent2.stops.index(available_values[0])
            child_vehicles[i] = parent2.vehicles[index_of_no_in_parent2]
            child_stops[i] = available_values.pop(0)

    return Chromosome(child_stops, child_vehicles)


# do the crossover, implemented according to the order crossover
def do_crossover(parent1: Chromosome, parent2: Chromosome):
    crossover_point_1 = random.randint(0, len(parent1.stops) - 1)
    crossover_point_2 = random.randint(0, len(parent1.stops) - 1)
    child_stops = [-1] * len(parent1.stops)
    used_values = []

    for i in range(min(crossover_point_1, crossover_point_2), max(crossover_point_1, crossover_point_2) + 1):
        child_stops[i] = parent1.stops[i]
        used_values.append(parent1.stops[i])

    available_values = [ele for ele in parent2.stops if ele not in used_values]

    for i in range(0, len(parent1.stops)):
        if child_stops[i] == -1:
            child_stops[i] = available_values.pop(0)

    child_1 = Chromosome(child_stops, parent1.vehicles.copy())
    child_2 = Chromosome(child_stops, parent2.vehicles.copy())

    evaluate_fitness(child_1)
    evaluate_fitness(child_2)

    if child_1.fitness > child_2.fitness:
        return child_1
    else:
        return child_2


# does the mutation by swapping to random elements
# TODO we could add other mutations and then select them randomly, see shift genes
# TODO add mutation rate
def do_mutation(c: Chromosome):
    if random.uniform(0, 1) < MUTATION_RATE:
        swap_gene(c)


# swaps two genes
def swap_gene(c: Chromosome):
    swapping_index_1 = random.randint(0, len(c.stops) - 1)
    swapping_index_2 = random.randint(0, len(c.stops) - 1)

    temp_stop = c.stops[swapping_index_1]
    c.stops[swapping_index_1] = c.stops[swapping_index_2]
    c.stops[swapping_index_2] = temp_stop

    swapping_index_1 = random.randint(0, len(c.stops) - 1)
    swapping_index_2 = random.randint(0, len(c.stops) - 1)

    temp_vehicle = c.vehicles[swapping_index_1]
    c.vehicles[swapping_index_1] = c.vehicles[swapping_index_2]
    c.vehicles[swapping_index_2] = temp_vehicle


# shuffles the genes
def shift_genes(c: Chromosome):
    pass


# shows the phenotype of a chromosome
def print_phenotype(c: Chromosome):
    path_costs, vehicle_weights = calculate_path_costs_and_weights(c)
    print(vehicle_weights)

    for i in range(0, NO_VEHICLES):
        print("Route #", i + 1, ":", sep="", end=" ")
        for j in range(0, len(c.vehicles)):
            if c.vehicles[j] == i:
                print(c.stops[j], end=" ")
        print("")

    print("The total costs of the path are:", "{:.2f}".format(sum(path_costs)))


# returns the best chromosome in a population
def get_best_chromosome(population):
    max_fitness = - sys.maxsize
    best_chrom = Chromosome([], [])
    for c in population:
        if c.fitness > max_fitness:
            max_fitness = c.fitness
            best_chrom = c
    return best_chrom


def ga_solve():
    curr_population = gen_population()
    for chrom in curr_population:
        evaluate_fitness(chrom)

    for i in range(0, NO_GENERATIONS):
        new_population = []
        for j in range(0, POPULATION_SIZE):
            parent1 = select_parent_roulette_selection(curr_population)

            if random.uniform(0, 1) < CROSSOVER_RATE:
                # TODO do we need to check that parent1 and parent2 are not the same or does that not matter?
                parent2 = select_parent_roulette_selection(curr_population)
                child = do_crossover(parent1, parent2)
            else:
                child = parent1

            do_mutation(child)
            evaluate_fitness(child)
            new_population.append(child)

        if KEEP_BEST:
            best = get_best_chromosome(curr_population)
            new_population[0] = best

        curr_population = new_population

    return get_best_chromosome(curr_population)


# calculates the distance based on euclidean metric measurement
def calc_dist(x1, y1, x2, y2):
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


# creates the distance matrix and the demands array
def calculate_map_context():
    file = open("data.csv")
    csvreader = csv.reader(file)
    next(csvreader)  # skip header
    rows = []
    for row in csvreader:
        rows.append(list(map(int, row)))

    _dist_matrix = []
    _demands = [0] * len(rows)
    for i in range(0, len(rows)):
        row = [0] * len(rows)
        _dist_matrix.append(row)

    for i in range(0, len(rows)):
        _demands[i] = rows[i][1]
        for j in range(i, len(rows)):
            start_x = rows[i][2]
            start_y = rows[i][3]
            end_x = rows[j][2]
            end_y = rows[j][3]
            dist = calc_dist(start_x, start_y, end_x, end_y)
            _dist_matrix[i][j] = dist
            _dist_matrix[j][i] = dist
    return _dist_matrix, _demands, rows


def plot_map(c: Chromosome, data):
    x_data = [d[2] for d in data]
    y_data = [d[3] for d in data]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:grey"]

    routes = []

    for i in range(0, NO_VEHICLES):
        route = [0]
        for j in range(0, len(c.vehicles)):
            if c.vehicles[j] == i:
                route.append(c.stops[j])
        route.append(0)
        routes.append(route)

    for i in range(0, len(routes)):
        x_points = []
        y_points = []
        for j in routes[i]:
            x_points.append(x_data[j])
            y_points.append(y_data[j])
        plt.plot(x_points[1:-1], y_points[1:-1], label="Route" + str(i + 1), marker='o', color=colors[i])
        plt.plot(x_points[:2], y_points[:2], color=colors[i], linestyle="--")
        plt.plot(x_points[-2:], y_points[-2:], color=colors[i], linestyle="--")

    plt.plot(x_data[0], y_data[0], marker='o', color='black')

    plt.legend()
    plt.show()


def check_costs_of_optimal_path():
    print("distance of his best solution")

    """
    Route1 = [4, 7, 42, 31, 20, 46, 26]
    Route2 = [36, 11, 15, 51, 2, 17, 14]
    Route3 = [37, 3, 34, 33, 21]
    Route4 = [1, 45, 6, 8]
    Route5 = [25, 41, 29]
    Route6 = [23, 52, 24, 44, 50, 48, 18]
    Route7 = [32, 38, 16, 40, 53, 5, 10, 12]
    Route8 = [30, 22, 19, 27, 13, 54, 28]
    Route9 = [47, 39, 49, 9, 35, 43]
    """

    Route1 = [0, 4, 7, 42, 31, 20, 46, 26, 0]
    Route2 = [0, 36, 11, 15, 51, 2, 17, 14, 0]
    Route3 = [0, 37, 3, 34, 33, 21, 0]
    Route4 = [0, 1, 45, 6, 8, 0]
    Route5 = [0, 25, 41, 29, 0]
    Route6 = [0, 23, 52, 24, 44, 50, 48, 18, 0]
    Route7 = [0, 32, 38, 16, 40, 53, 5, 10, 12, 0]
    Route8 = [0, 30, 22, 19, 27, 13, 54, 28, 0]
    Route9 = [0, 47, 39, 49, 9, 35, 43, 0]

    paths = [Route1, Route2, Route3, Route4, Route5, Route6, Route7, Route8, Route9]
    costs = []

    for p in paths:
        p_cost = 0
        for i in range(1, len(p)):
            p_cost += distance_matrix[i][i - 1]
        costs.append(p_cost)
    print(sum(costs))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    distance_matrix, demands, data_matrix = calculate_map_context()

    start_time = time.time()
    best_chromosome = ga_solve()
    end_time = time.time()

    print_phenotype(best_chromosome)
    print("Runtime of the algorithm:", "{:.2f}".format(end_time - start_time))
    plot_map(best_chromosome, data_matrix)

    # check_costs_of_optimal_path()

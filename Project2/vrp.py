# This is a sample Python script.

import csv
import math
import random
from dataclasses import dataclass, field

from typing import List

NO_GENERATIONS = 10
POPULATION_SIZE = 10
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.1
OVER_WEIGHT_PENALTY = 1000
SELECTION_PRESSURE = 1.5

@dataclass
class Chromosome:
    stops: List[int] = field(default_factory=list)
    vehicles: List[int] = field(default_factory=list)
    fitness: int = 0


def calc_dist(x1, y1, x2, y2):
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


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


def fun_fitness(costs, weight):
    if weight > 100:
        weight_penalty = (weight - 100) * OVER_WEIGHT_PENALTY
    else:
        weight_penalty = 0
    fitness = costs + weight_penalty
    return fitness

def get_fitness(c: Chromosome):
    path_costs = [0] * 9
    vehicle_weight = [0] * 9
    prev_stop = [0] * 9

    for i in range(0, len(c.vehicles)):
        stop = c.stops[i] # the current stop
        vehicle_no = c.vehicles[i] # the current driver that stops for this customer
        dist = distance_matrix[prev_stop[vehicle_no]][stop] # distance driver makes for this customer
        path_costs[vehicle_no] += dist
        vehicle_weight[vehicle_no] += demands[stop]
        prev_stop[vehicle_no] = stop

    # calculate costs for return to depot
    for i in range(0, len(prev_stop)):
        return_dist = distance_matrix[prev_stop[i]][0]
        path_costs[i] += return_dist

    total_fitness = 0
    for i in range(0, len(path_costs)):
        f = fun_fitness(path_costs[i], vehicle_weight[i])
        total_fitness += f

    c.fitness = 1 / total_fitness

def get_rank(c: Chromosome, total_fitness: int):

    # original ranks from original fitness
    # total sum fitness
    # scale individual fitness
    # sum of scaled fitness
    # select based on rank
    return


def select_parent(chromosomes):
    ranks = []
    total_fitness = 0
    for chrom in chromosomes:
        total_fitness += chrom.fitness

    for chrom in chromosomes:
        get_rank(chrom)


def ga_solve():
    curr_population = gen_population()
    for chrom in curr_population:
        get_fitness(chrom)
    for i in range(0, NO_GENERATIONS):
        for j in range(0, POPULATION_SIZE):
            parent1 = select_parent(curr_population)


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
    return _dist_matrix, _demands


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    distance_matrix, demands = calculate_map_context()
    ga_solve()

from ast import main
from dataclasses import dataclass
from typing import List
import csv
import os


@dataclass
class Node:
    state: int
    cost: int
    path: List[int]

def rec_dist(x_i: int, x_f: int, y_i: int, y_f: int) -> int:
    return abs(x_i - x_f) + abs(y_i - y_f)

def select_best_node():

def expand():

def solve():    

def main():
    #i = input("Starting node index: \n")
    #f = input("Ending node index: \n")
    #w = input("Weight parameter: \n")

    i = 1
    f = 10
    w = 0.5

    file = open("100_nodes.csv")
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
            rows.append(list(map(int, row)))

    start_row = rows[i]
    end_row = rows[f]

    start_node = Node(start_row[0], 0, [])
    frontier = List[Node]
    frontier.append(start_node)




if __name__ == '__main__':
    main()



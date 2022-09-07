from ast import main
from dataclasses import dataclass
from typing import List
import csv

@dataclass
class Node:
    x: int
    y: int
    state: int # matrix index
    cost: int
    path: List[int]

def rec_dist(x_i: int, x_f: int, y_i: int, y_f: int) -> int:
    return abs(x_i - x_f) + abs(y_i - y_f)

def select_best_node(frontier, w: float, goal_node: Node) -> Node:
    fbest: int = float('inf')
    ibest: int = -1
    for i in range(len(frontier)):
        node = frontier[i]
        heuristic_cost = rec_dist(node.x, goal_node.x, node.y, goal_node.y)
        f = w * node.cost + (1 - w)* heuristic_cost

        if f < fbest:
            fbest = f
            ibest = i

    res_node = frontier[ibest]
    frontier.remove(res_node)
    return res_node     

def search_node(nodes, target: Node) -> Node:
    for node in nodes:
        if(node.state == target.state):
            return 1 
    return -1

def expand(matrix: List[List[int]], curr_node: Node):
    children = []
    curr_row = matrix[curr_node.state]
    for i in range(3, len(curr_row)):
        if(curr_row[i] != 0):
            child_row = matrix[i]
            child_costs = curr_node.cost + curr_row[i]
            child_path = curr_node.path.append(i)
            child = Node(child_row[1], child_row[2], child_row[0], child_costs, child_path)
            children.append(child)
    return children

def astar_search(matrix: List[List[int]], w: float, start_node: Node, goal_node: Node):
    frontier = []
    frontier.append(start_node)
    nodes_generated: int = 0;

    while len(frontier) != 0:
        node = select_best_node(frontier, w, goal_node)
        if (node.state == goal_node.state):
            return node, nodes_generated
        children = expand(matrix, node)
        nodes_generated += len(children)
        for child in children:
            i = search_node(frontier, child)
            if (i == -1):
                frontier.append(child)
            elif (child.cost < frontier[i].cost):
                frontier[i] = child
    return None, nodes_generated

def main():
    #i = input("Starting node index: \n")
    #f = input("Ending node index: \n")
    #w = input("Weight parameter: \n")

    i: int = 0
    f: int = 1
    w: float = 0.5

    file = open("100_nodes.csv")
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
            rows.append(list(map(int, row)))

    start_row = rows[i]
    end_row = rows[f]
    start_node = Node(start_row[1], start_row[2], start_row[0], 0, [])
    goal_node = Node(end_row[1], end_row[2], end_row[0], 0, [])
    
    res, nodes_generated = astar_search(rows, w, start_node, goal_node)

    if(res is None):
        print("Failure, no path found\n Nodes generated: ", nodes_generated)
    else: 
        print("Path Length: ", len(res.path), "\nPath: ", res.path, "\nNodes generated: ", nodes_generated)

if __name__ == '__main__':
    main()



from python_tsp.exact import solve_tsp_dynamic_programming
import numpy as np
import math


def create_distance_matrix(nodes):
    distance_matrix = np.zeros((len(nodes), len(nodes)))
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if j >= i:
                break
            distance_matrix[i, j] = _euclidean_distance(n1, n2)
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix


def _euclidean_distance(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def solve(nodes):
    distances = create_distance_matrix(nodes)
    permutation, distance = solve_tsp_dynamic_programming(distances)
    return distance



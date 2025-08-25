#!/usr/bin/env python
# Created by "Thieu" at 13:57, 07/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# In the context of the Mealpy for the Traveling Salesman Problem (TSP), a solution is a possible route that
# represents a tour of visiting all the cities exactly once and returning to the starting city. The solution is typically
# represented as a permutation of the cities, where each city appears exactly once in the permutation.

# For example, let's consider a TSP instance with 5 cities labeled as A, B, C, D, and E. A possible solution could be
# represented as the permutation [A, B, D, E, C], which indicates the order in which the cities are visited. This solution
# suggests that the tour starts at city A, then moves to city B, then D, E, and finally C before returning to city A.


import numpy as np
from mealpy import PermutationVar, WOA, Problem

# Define the positions of the cities
city_positions = np.array([[60, 200], [180, 200], [80, 180], [140, 180], [20, 160], [100, 160], [200, 160], [140, 140], [40, 120], [100, 120], [180, 100], [60, 80], [120, 80], [180, 60], [20, 40], [100, 40], [200, 40], [20, 20], [60, 20], [160, 20]])
city_positions = np.array([[38, 83], [54, 13], [19, 2], [96, 30], [52, 7], [48, 32], [53, 92], [70, 6], [92, 27], [96, 63], [8, 56], [67, 100], [59, 79], [48, 69], [35, 79], [64, 44], [14, 79], [2, 8], [1, 6], [47, 95]])
city_positions = np.array([(14, 17), (69, 76), (76, 17), (81, 52), (63, 53), (14, 77), (33, 27), (93, 80), (16, 50), (52, 99), (39, 36), (54, 36), (28, 74), (39, 47), (37, 29), (14, 65), (30, 2), (5, 6), (25, 16), (98, 32), (55, 42), (12, 86), (13, 94), (4, 26), (87, 44), (48, 17), (38, 93), (39, 63), (9, 72), (0, 94), (78, 83), (79, 66), (67, 21), (73, 5), (3, 32), (32, 7), (93, 93), (96, 77), (43, 71), (15, 46), (97, 51), (51, 24), (1, 26), (86, 86), (3, 85), (54, 25), (15, 90), (49, 12), (82, 96), (82, 12), (34, 9), (70, 6), (93, 78), (98, 88), (41, 32), (88, 18), (57, 21), (91, 69), (58, 17), (7, 33), (25, 61), (57, 25), (17, 55), (92, 81), (36, 14), (17, 45), (17, 25), (3, 83), (82, 29), (82, 82), (84, 28), (99, 6), (6, 33), (7, 32), (46, 67), (24, 71), (24, 14), (42, 93), (3, 98), (98, 34), (33, 95), (82, 45), (13, 52), (66, 69), (54, 41), (26, 80), (31, 58), (87, 51), (36, 60), (95, 46), (11, 0), (98, 50), (37, 76), (24, 92), (53, 96), (14, 35), (35, 25), (3, 74), (49, 53), (0, 77)])

num_cities = len(city_positions)
data = {
    "city_positions": city_positions,
    "num_cities": num_cities,
}

class TspProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        super().__init__(bounds, minmax, **kwargs)
        self.data = data

    @staticmethod
    def calculate_distance(city_a, city_b):
        # Calculate Euclidean distance between two cities
        return np.linalg.norm(city_a - city_b)

    @staticmethod
    def calculate_total_distance(route, city_positions):
        # Calculate total distance of a route
        total_distance = 0
        num_cities = len(route)
        for idx in range(num_cities):
            current_city = route[idx]
            next_city = route[(idx + 1) % num_cities]  # Wrap around to the first city
            total_distance += TspProblem.calculate_distance(city_positions[current_city], city_positions[next_city])
        return total_distance

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        route = x_decoded["per_var"]
        fitness = self.calculate_total_distance(route, self.data["city_positions"])
        return fitness


bounds = PermutationVar(valid_set=list(range(0, num_cities)), name="per_var")
problem = TspProblem(bounds=bounds, minmax="min", data=data)

model = WOA.OriginalWOA(epoch=100, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")      # Decoded (Real) solution
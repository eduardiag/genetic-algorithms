import numpy as np
from math import sqrt
from matplotlib import pyplot
from mealpy import Problem, PermutationVar
from mealpy.evolutionary_based.GA import BaseGA

# Parameters
n_cities = 20

print("Running TSP with MEALPY")
print(f" - Number of cities: {n_cities}")

# Sample points
sample_points = [(38, 83), (54, 13), (19, 2), (96, 30), (52, 7), (48, 32), (53, 92),
                 (70, 6), (92, 27), (96, 63), (8, 56), (67, 100), (59, 79), (48, 69),
                 (35, 79), (64, 44), (14, 79), (2, 8), (1, 6), (47, 95)]

# Distance matrix
mat = np.zeros((n_cities, n_cities))
for i1 in range(n_cities):
    for i2 in range(n_cities):
        mat[i1][i2] = sqrt((sample_points[i1][0] - sample_points[i2][0])**2 +
                           (sample_points[i1][1] - sample_points[i2][1])**2)
dist = np.array(mat)

class TSPProblem(Problem):
    def __init__(self, dist_matrix):
        self.dist = dist_matrix
        dim = dist_matrix.shape[0]
        
        # Define permutation variable
        var = PermutationVar(valid_set=list(range(dim)), name="route")
        bounds = [var]
        
        super().__init__(bounds=bounds, minmax="min", name="TSP")
    
    def obj_func(self, solution):
        route = solution  # solution is already the permutation array
        total_dist = 0
        for i in range(len(route)):
            from_city = int(route[i])  # Ensure integer index
            to_city = int(route[(i + 1) % len(route)])  # Ensure integer index
            total_dist += self.dist[from_city, to_city]
        return total_dist



problem = TSPProblem(dist_matrix=dist)
model = BaseGA(epoch=1000, pop_size=200)  # Increased for better results
agent = model.solve(problem)

print("Best route:", agent.solution)
print("Distance:", agent.target.fitness)

# Visualization
route = agent.solution
x_coords = [sample_points[point][0] for point in route] + [sample_points[route[0]][0]]
y_coords = [sample_points[point][1] for point in route] + [sample_points[route[0]][1]]

pyplot.figure(figsize=(8, 6))
pyplot.plot(x_coords, y_coords, marker='o', linestyle='-')
for p in route:
    pyplot.text(sample_points[p][0], sample_points[p][1], str(p), fontsize=9)

pyplot.title(f'TSP Solution (Distance: {agent.target.fitness:.2f})')
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.grid(True)
pyplot.axis('equal')
pyplot.show()
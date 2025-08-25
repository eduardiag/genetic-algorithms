from math import sqrt
from mealpy import PermutationVar, Problem, GA
import matplotlib.pyplot as plt
import random
import numpy

# Parameters
n_cities = 300
dim_grid = 100
POP_SIZE = 200
GENS = 10
ELITE_SIZE = POP_SIZE // 20

# Generació de punts aleatoris
sample_points = [(random.randint(0, dim_grid), random.randint(0, dim_grid)) for _ in range(n_cities)]

print("Running TSP with parameters:")
print(f" - Number of cities: {n_cities}")
print(f" - Grid size: {dim_grid}")
print(f" - Population size: {POP_SIZE}")
print(f" - Generations: {GENS}")




dist = numpy.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        dx = sample_points[i][0] - sample_points[j][0]
        dy = sample_points[i][1] - sample_points[j][1]
        dist[i][j] = sqrt(dx * dx + dy * dy)

def tour_length(tour):
    total = 0.0
    for i in range(len(tour)):
        a = int(tour[i])
        b = int(tour[(i + 1) % len(tour)])
        total += dist[a, b]
    return total

def two_opt(tour, max_passes=10):
    best = list(map(int, tour))
    best_len = tour_length(best)
    n = len(best)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = best[i - 1], best[i]
                c, d = best[k], best[(k + 1) % n]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < -1e-9:
                    best[i:k + 1] = reversed(best[i:k + 1])
                    best_len += delta
                    improved = True
        if not improved:
            break
    return best, best_len

def tsp_objective_function(solution):
    total = 0.0
    for i in range(len(solution)):
        a = int(solution[i])
        b = int(solution[(i + 1) % len(solution)])
        total += dist[a, b]
    return total

print("Running TSP with MEALPY Genetic Algorithm...")
bounds = PermutationVar(valid_set=list(range(n_cities)), name="tour")
problem = Problem(bounds=bounds, minmax="min", obj_func=tsp_objective_function, name="TSP")
model = GA.BaseGA(epoch=GENS, pop_size=POP_SIZE, pc=0.9, pm=0.2)
best_agent = model.solve(problem)

ga_sol = [int(x) for x in best_agent.solution]
loc_sol, loc_len = two_opt(ga_sol, max_passes=10)
best_distance = min(best_agent.target.fitness, loc_len)
best_route = loc_sol if loc_len < best_agent.target.fitness else ga_sol

print(f"Best distance: {best_distance:.2f}")
print(f"Route (primeres 20): {best_route[:20]}...")

x = [sample_points[i][0] for i in best_route] + [sample_points[best_route[0]][0]]
y = [sample_points[i][1] for i in best_route] + [sample_points[best_route[0]][1]]

plt.figure(figsize=(8, 8))
plt.plot(x, y, 'b-', linewidth=2, alpha=0.7, label='Tour')
plt.scatter([p[0] for p in sample_points], [p[1] for p in sample_points], c='red', s=50, zorder=5)
plt.scatter(x[0], y[0], c='green', s=100, marker='s', zorder=6, label='Start')
plt.title(f'TSP Solution - Distance: {best_distance:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
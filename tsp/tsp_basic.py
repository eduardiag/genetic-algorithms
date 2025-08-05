import random
import numpy as np
from math import sqrt
from matplotlib import pyplot

n_cities = 20
dim_grid = 100
POP_SIZE = 100
GENS = 100
MUTATION_RATE = 0.1

print("Running TSP with parameters:")
print(f" - Number of cities: {n_cities}")
print(f" - Grid size: {dim_grid}")
print(f" - Population size: {POP_SIZE}")
print(f" - Generations: {GENS}")
print(f" - Mutation rate: {MUTATION_RATE}")

sample_points = [(random.randint(0, dim_grid), random.randint(0, dim_grid)) for _ in range(n_cities)]

sample_points = [(38, 83), (54, 13), (19, 2), (96, 30), (52, 7), (48, 32), (53, 92), (70, 6), (92, 27), 
(96, 63), (8, 56), (67, 100), (59, 79), (48, 69), (35, 79), (64, 44), (14, 79), (2, 8), 
(1, 6), (47, 95)]

mat = np.zeros((n_cities, n_cities))
for i1 in range(n_cities):
    for i2 in range(n_cities):
        dx = sample_points[i1][0] - sample_points[i2][0]
        dy = sample_points[i1][1] - sample_points[i2][1]
        mat[i1][i2] = sqrt(dx * dx + dy * dy)
dist = np.array(mat)



def create_individual():
    route = list(range(n_cities))
    random.shuffle(route)
    return route

def fitness(route):
    return -sum(dist[route[i], route[(i + 1) % n_cities]] for i in range(n_cities))

def selection(population):
    return sorted(population, key=fitness, reverse=True)[:2]

def crossover(p1, p2):
    start, end = sorted(random.sample(range(n_cities), 2))
    child = [None] * n_cities
    child[start:end] = p1[start:end]
    fill = [city for city in p2 if city not in child]
    idx = 0
    for i in range(n_cities):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child

def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(n_cities), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual



population = [create_individual() for _ in range(POP_SIZE)]

for gen in range(GENS):
    population = [mutate(crossover(*selection(population))) for _ in range(POP_SIZE)]

best = max(population, key=fitness)
print("Sample points:", sample_points)
print("Best route:", best)
print("Distance:", -fitness(best))


# --------------------------------------

# Visualització del millor recorregut
x_coords = [sample_points[point][0] for point in best] + [sample_points[best[0]][0]]
y_coords = [sample_points[point][1] for point in best] + [sample_points[best[0]][1]]

pyplot.figure(figsize=(8, 6))
pyplot.plot(x_coords, y_coords, marker='o', linestyle='-')
for p in best:
    pyplot.text(sample_points[p][0], sample_points[p][1], p, fontsize=9)

pyplot.title(f'Tour aleatori de {n_cities} punts enters')
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.grid(True)
pyplot.axis('equal')
pyplot.show()

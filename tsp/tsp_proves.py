import random
import numpy as np
from math import sqrt
from matplotlib import pyplot




def create_individual():
    route = list(range(n_cities))
    random.shuffle(route)
    return route

def fitness(route):
    return -sum(dist[route[i], route[(i + 1) % n_cities]] for i in range(n_cities))

def selection(population, k):
    selected = random.sample(population, k)
    return sorted(selected, key=fitness, reverse=True)[:2]

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
        i, j = sorted(random.sample(range(n_cities), 2))
        individual[i:j] = reversed(individual[i:j])
    return individual




# Problema
sample_points = [(38, 83), (54, 13), (19, 2), (96, 30), (52, 7), (48, 32), (53, 92), (70, 6), (92, 27), (96, 63), (8, 56), (67, 100), (59, 79), (48, 69), (35, 79), (64, 44), (14, 79), (2, 8), (1, 6), (47, 95)]
sample_points = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160), (100, 160), (200, 160), (140, 140), (40, 120), (100, 120), (180, 100), (60, 80), (120, 80), (180, 60), (20, 40), (100, 40), (200, 40), (20, 20), (60, 20), (160, 20)]
sample_points = [(14, 17), (69, 76), (76, 17), (81, 52), (63, 53), (14, 77), (33, 27), (93, 80), (16, 50), (52, 99), (39, 36), (54, 36), (28, 74), (39, 47), (37, 29), (14, 65), (30, 2), (5, 6), (25, 16), (98, 32), (55, 42), (12, 86), (13, 94), (4, 26), (87, 44), (48, 17), (38, 93), (39, 63), (9, 72), (0, 94), (78, 83), (79, 66), (67, 21), (73, 5), (3, 32), (32, 7), (93, 93), (96, 77), (43, 71), (15, 46), (97, 51), (51, 24), (1, 26), (86, 86), (3, 85), (54, 25), (15, 90), (49, 12), (82, 96), (82, 12), (34, 9), (70, 6), (93, 78), (98, 88), (41, 32), (88, 18), (57, 21), (91, 69), (58, 17), (7, 33), (25, 61), (57, 25), (17, 55), (92, 81), (36, 14), (17, 45), (17, 25), (3, 83), (82, 29), (82, 82), (84, 28), (99, 6), (6, 33), (7, 32), (46, 67), (24, 71), (24, 14), (42, 93), (3, 98), (98, 34), (33, 95), (82, 45), (13, 52), (66, 69), (54, 41), (26, 80), (31, 58), (87, 51), (36, 60), (95, 46), (11, 0), (98, 50), (37, 76), (24, 92), (53, 96), (14, 35), (35, 25), (3, 74), (49, 53), (0, 77)]

# Paràmetres
POP_SIZE = 100
GENS = 1200
MUTATION_RATE = 0.5
n_experiments = 10
dim_grid = 100

# Matriu de distàncies
n_cities = len(sample_points)
mat = np.zeros((n_cities, n_cities))
for i1 in range(n_cities):
    for i2 in range(n_cities):
        mat[i1][i2] = sqrt((sample_points[i1][0] - sample_points[i2][0]) ** 2 + (sample_points[i1][1] - sample_points[i2][1]) ** 2)
dist = np.array(mat)



print("Running TSP with parameters:")
print(f" - Number of cities: {n_cities}")
print(f" - Grid size: {dim_grid}")
print(f" - Population size: {POP_SIZE}")
print(f" - Generations: {GENS}")
print(f" - Mutation rate: {MUTATION_RATE}")
print("-----------------")



top_route = list(range(n_cities))
top_distance = -fitness(top_route)
conv = []

for exp in range(n_experiments):

    print("Running experiment", exp)

    population = [create_individual() for _ in range(POP_SIZE)]

    seq = []
    for gen in range(GENS):
        # Elitisme: guarda el millor
        new_population = [max(population, key=fitness)]
        for _ in range(POP_SIZE - 1):
            parents = selection(population, POP_SIZE // 4)
            child = crossover(*parents)
            child = mutate(child)
            new_population.append(child)
        population = new_population
        seq.append(-fitness(max(population, key=fitness)))
        if gen % 100 == 0:
            print(f"Generation {gen}: Best distance = {seq[-1]}")    
    conv.append(seq)

    best = max(population, key=fitness)
    distance = -fitness(best)

    print("Best route:", best)
    print("Distance:", distance)
    print("---------------")
    if distance < top_distance:
        top_route = best
        top_distance = distance

print("BEST OVERALL ROUTE:", top_route)
print("BEST OVERALL DISTANCE:", top_distance)

pyplot.figure(figsize=(10, 6))
for seq in conv:
    pyplot.plot(seq)
pyplot.title("Convergence curves")
pyplot.xlabel("Iteration")
pyplot.ylabel("Best fitness")
pyplot.grid(True)
pyplot.tight_layout()
pyplot.show()
import random
import numpy as np
from math import sqrt
from matplotlib import pyplot


# Algorisme genètic per al TSP (Travelling Salesman Problem)


# Parameters
n_cities = 20
dim_grid = 100
POP_SIZE = 100
GENS = 100
MUTATION_RATE = 0.1
ELITE_SIZE = POP_SIZE // 20

print("Running TSP with parameters:")
print(f" - Number of cities: {n_cities}")
print(f" - Grid size: {dim_grid}")
print(f" - Population size: {POP_SIZE}")
print(f" - Generations: {GENS}")
print(f" - Mutation rate: {MUTATION_RATE}")



# Generació de punts aleatoris
sample_points = [(random.randint(0, dim_grid), random.randint(0, dim_grid)) for _ in range(n_cities)]

# Mostra preestablerta
sample_points = [(38, 83), (54, 13), (19, 2), (96, 30), (52, 7), (48, 32), (53, 92), (70, 6), (92, 27), (96, 63), (8, 56), (67, 100), (59, 79), (48, 69), (35, 79), (64, 44), (14, 79), (2, 8), (1, 6), (47, 95)]

mat = np.zeros((n_cities, n_cities))
for i1 in range(n_cities):
    for i2 in range(n_cities):
        dx = sample_points[i1][0] - sample_points[i2][0]
        dy = sample_points[i1][1] - sample_points[i2][1]
        mat[i1][i2] = sqrt(dx * dx + dy * dy)
dist = np.array(mat)



# Funcions de l'algorisme genètic
def create_individual():
    route = list(range(n_cities))
    random.shuffle(route)
    return route

def fitness(route):
    return -sum(dist[route[i], route[(i + 1) % n_cities]] for i in range(n_cities))

"""
def selection(population):
    return sorted(population, key=fitness, reverse=True)[:2]

def selection(population):
    # Selecció per ruleta: probabilitat proporcional al fitness.
    fitness_values = np.array([fitness(ind) for ind in population])
    
    # Com que els fitness són negatius (distància negativa), els invertim
    adjusted_fitness = fitness_values - np.min(fitness_values) + 1e-6  # afegim petit valor per evitar zeros

    # Normalitzem les probabilitats
    probabilities = adjusted_fitness / np.sum(adjusted_fitness)

    # Seleccionem dos individus segons aquestes probabilitats
    selected_indices = np.random.choice(len(population), size=2, p=probabilities)
    return [population[selected_indices[0]], population[selected_indices[1]]]
"""

def selection(population, k):
    selected = random.sample(population, k)
    return sorted(selected, key=fitness, reverse=True)[:2]

def crossover(p1, p2):
    start, end = sorted(random.sample(range(len(p1)), 2))
    child = [None] * len(p1)
    child[start:end] = p1[start:end]
    existing = set(child[start:end])
    idx = 0
    for city in p2:
        if city not in existing:
            while child[idx] is not None:
                idx += 1
            child[idx] = city
    return child

"""
def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(n_cities), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual
"""

def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(n_cities), 2))
        individual[i:j] = reversed(individual[i:j])
    return individual


# Main loop de l'algorisme genètic
population = [create_individual() for _ in range(POP_SIZE)]

for gen in range(GENS):
    # population = [mutate(crossover(*selection(population))) for _ in range(POP_SIZE)]
    # Elitisme: guarda el millor a cada generació
    population = sorted(population, key=fitness, reverse=True)[:ELITE_SIZE] + [mutate(crossover(*selection(population, POP_SIZE // 4))) for _ in range(POP_SIZE - ELITE_SIZE)]
    # print(-fitness(max(population, key=fitness)))

best = max(population, key=fitness)
print("Sample points:", sample_points)
print("Best route:", best)
print("Distance:", -fitness(best))



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

import random
import numpy as np
from math import sqrt
from matplotlib import pyplot



# Parameters
n_cities = 100
dim_grid = 100
POP_SIZE = 200
GENS = 1000
MUTATION_RATE = 0.3
ELITE_SIZE = POP_SIZE // 20

# Generació de punts aleatoris
sample_points = [(random.randint(0, dim_grid), random.randint(0, dim_grid)) for _ in range(n_cities)]

print("Running TSP with parameters:")
print(f" - Number of cities: {n_cities}")
print(f" - Grid size: {dim_grid}")
print(f" - Population size: {POP_SIZE}")
print(f" - Generations: {GENS}")
print(f" - Mutation rate: {MUTATION_RATE}")





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

def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(n_cities), 2))
        individual[i:j] = reversed(individual[i:j])
    return individual


# Main loop de l'algorisme genètic
population = [create_individual() for _ in range(POP_SIZE)]

pyplot.ion()  # Mode interactiu activat
fig, ax = pyplot.subplots(figsize=(8, 6))  # Crear figura i eixos

for gen in range(GENS):
    # Selecció + creació de nova població amb elitisme
    population = sorted(population, key=fitness, reverse=True)[:ELITE_SIZE] + \
                 [mutate(crossover(*selection(population, POP_SIZE // 4))) for _ in range(POP_SIZE - ELITE_SIZE)]

    best = max(population, key=fitness)

    if gen % 10 == 0:  # Actualitzar gràfic cada 10 generacions
        print(f"Generation {gen}: Best fitness = {-fitness(best)}")
            
        # Dibuixar millor recorregut
        ax.cla()  # Neteja eixos per actualitzar el gràfic

        x_coords = [sample_points[point][0] for point in best] + [sample_points[best[0]][0]]
        y_coords = [sample_points[point][1] for point in best] + [sample_points[best[0]][1]]

        ax.plot(x_coords, y_coords, marker='o', linestyle='-')

        for p in best:
            ax.text(sample_points[p][0], sample_points[p][1], str(p), fontsize=9)

        ax.set_title(f'Generació {gen} - Millor recorregut')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        ax.axis('equal')

        pyplot.pause(0.3)  # Pausa per actualitzar el gràfic

# Desactivar mode interactiu i mostrar el resultat final
pyplot.ioff()
pyplot.show()
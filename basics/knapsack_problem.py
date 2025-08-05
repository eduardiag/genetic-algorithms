import random

# ----------------------------------------------------------

# Exemple de dades
items = [
    {"value": 10, "weight": 5},
    {"value": 40, "weight": 4},
    {"value": 30, "weight": 6},
    {"value": 50, "weight": 3},
    {"value": 35, "weight": 2},
]

MAX_WEIGHT = 10
POP_SIZE = 100
GENS = 100
MUTATION_RATE = 0.05

# ----------------------------------------------------------


def create_individual(n_items):
    return [random.randint(0, 1) for _ in range(n_items)]

def fitness(individual):
    total_value = total_weight = 0
    for gene, item in zip(individual, items):
        if gene:
            total_value += item["value"]
            total_weight += item["weight"]
    if total_weight > MAX_WEIGHT:
        return 0  # Penalització dura
    return total_value

def selection(population):
    return sorted(population, key=fitness, reverse=True)[:2]

def crossover(p1, p2):
    point = random.randint(1, len(p1) - 2)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate(individual):
    return [gene if random.random() > MUTATION_RATE else 1 - gene for gene in individual]

# Main loop
population = [create_individual(len(items)) for _ in range(POP_SIZE)]

for gen in range(GENS):
    new_population = []
    for _ in range(POP_SIZE // 2):
        parents = selection(population)
        child1, child2 = crossover(*parents)
        new_population.extend([mutate(child1), mutate(child2)])
    population = new_population

best = max(population, key=fitness)
print("Best solution:", best)
print("Total value:", fitness(best))

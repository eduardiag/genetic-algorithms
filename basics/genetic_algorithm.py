import random

# --- CONFIGURACIÓ ---
NUM_BITS = 5                # Codificació binària de x
POP_SIZE = 10               # Nombre d'individus
GENERATIONS = 50            # Iteracions
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01
ELITISM = True              # Manté el millor individu



# --- FITNESS ---
def decode(individual):
    """Converteix una llista binària a enter"""
    return int("".join(map(str, individual)), 2)

def fitness(individual):
    """Fitness = x^2"""
    x = decode(individual)
    return x ** 2

# --- OPERADORS GENÈTICS ---
def initial_population():
    return [[random.randint(0, 1) for _ in range(NUM_BITS)] for _ in range(POP_SIZE)]

def tournament_selection(population, k=3):
    contenders = random.sample(population, k)
    return max(contenders, key=fitness)

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, NUM_BITS - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1[:], parent2[:]

def mutate(individual):
    return [bit if random.random() > MUTATION_RATE else 1 - bit for bit in individual]

# --- GA PRINCIPAL ---
def genetic_algorithm():
    population = initial_population()
    best = max(population, key=fitness)

    for gen in range(GENERATIONS):
        new_population = []

        if ELITISM:
            new_population.append(best[:])

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(child2))

        population = new_population
        current_best = max(population, key=fitness)
        if fitness(current_best) > fitness(best):
            best = current_best

        # Imprimir l'evolució
        print(f"Gen {gen:2d} | Best x = {decode(best):2d} | f(x) = {fitness(best)}")

    return best




# --- EXECUCIÓ ---
if __name__ == "__main__":
    best_ind = genetic_algorithm()
    print("\nMillor solució trobada:")
    print(f"x = {decode(best_ind)}, f(x) = {fitness(best_ind)}")

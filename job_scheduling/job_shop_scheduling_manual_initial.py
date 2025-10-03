import random
from collections import Counter


# Genera una llista de jobs aleatoris
def generate_random_jobs(num_jobs, num_machines, operations_per_job, min_duration=1, max_duration=10):
    jobs = []
    
    if isinstance(operations_per_job, list):
        if len(operations_per_job) != num_jobs:
            raise ValueError("La llista operations_per_job ha de tenir num_jobs elements.")
        op_counts = operations_per_job
    else:
        op_counts = [operations_per_job] * num_jobs
    
    for job_idx in range(num_jobs):
        job = []
        for _ in range(op_counts[job_idx]):
            machine_id = random.randint(0, num_machines - 1)
            duration = random.randint(min_duration, max_duration)
            job.append((machine_id, duration))
        jobs.append(job)
    
    return jobs

# Exemple de jobs senzill:
jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 2)],
]

# Generació aleatòria de jobs: llista de llistes d'operacions (machine_id, duration)
jobs = generate_random_jobs(20, 10, 5, 1, 10)

# Exemple de jobs complexos: llista de llistes d'operacions (machine_id, duration)
# Aquesta llista s'ha generat usant generate_random_jobs(20, 10, 5, 1, 10)
jobs = [[(9, 2), (7, 2), (2, 6), (8, 2), (1, 9)], [(4, 5), (8, 6), (5, 5), (5, 3), (2, 10)], [(4, 10), (5, 6), (1, 3), (3, 1), (9, 8)], [(9, 5), (6, 3), (3, 8), (2, 9), (0, 2)], [(1, 4), (4, 5), (7, 4), (3, 8), (2, 1)], [(6, 8), (9, 4), (1, 5), (4, 10), (5, 9)], [(8, 10), (1, 10), (7, 10), (9, 4), (0, 8)], [(6, 6), (6, 8), (2, 9), (4, 1), (4, 4)], [(8, 9), (2, 10), (4, 7), (6, 3), (3, 1)], [(6, 1), (5, 5), (3, 2), (1, 8), (0, 7)], [(7, 8), (3, 7), (3, 1), (0, 10), (7, 1)], [(8, 4), (8, 7), (5, 9), (7, 5), (2, 4)], [(5, 8), (1, 2), (8, 1), (8, 4), (4, 6)], [(3, 8), (8, 9), (5, 6), (4, 10), (4, 1)], [(6, 7), (0, 8), (9, 6), (1, 3), (1, 6)], [(9, 9), (3, 2), (5, 10), (2, 3), (4, 7)], [(6, 4), (7, 9), (0, 10), (5, 4), (9, 2)], [(3, 7), (3, 9), (9, 5), (2, 3), (2, 10)], [(7, 3), (9, 5), (0, 6), (5, 2), (2, 8)], [(2, 7), (4, 2), (9, 1), (9, 5), (7, 2)]]


print("Jobs:", jobs)

num_jobs = len(jobs)
num_operations = sum(len(job) for job in jobs)
num_machines = max(max(op[0] for op in job) for job in jobs) + 1
job_counts = [len(job) for job in jobs]





# Genera un cromosoma aleatori: seqüència de job IDs amb repeticions
def generate_chromosome():
    gene = []
    for job_id, count in enumerate(job_counts):
        gene += [job_id] * count
    random.shuffle(gene)
    return gene

# Decodifica un cromosoma i construeix el schedule: retorna makespan
def evaluate_chromosome(chromosome):
    job_next_op = [0] * num_jobs
    machine_available = [0] * num_machines
    job_available = [0] * num_jobs

    schedule = []

    for job_id in chromosome:
        op_idx = job_next_op[job_id]
        machine, duration = jobs[job_id][op_idx]
        
        start_time = max(machine_available[machine], job_available[job_id])
        end_time = start_time + duration
        
        machine_available[machine] = end_time
        job_available[job_id] = end_time
        job_next_op[job_id] += 1
        
        schedule.append((job_id, op_idx, machine, start_time, end_time))

    makespan = max(end for *_, end in schedule)
    return makespan, schedule

# Selecció: torneig binari
def tournament_selection(pop, k=2):
    return min(random.sample(pop, k), key=lambda c: evaluate_chromosome(c)[0])

# Crossover d’ordre (Order Crossover)
def crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    a, b = sorted(random.sample(range(size), 2))
    child[a:b] = parent1[a:b]
    job_counts = Counter(parent1)
    child_counts = Counter(child[a:b])

    j = 0
    for i in range(size):
        if child[i] is None:
            while True:
                job = parent2[j]
                if child_counts[job] < job_counts[job]:
                    child[i] = job
                    child_counts[job] += 1
                    j += 1
                    break
                j += 1

    return child

# Mutació: swap aleatori
def mutate(chromosome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# Algorisme genètic
def genetic_algorithm(generations=100, pop_size=50):
    population = [generate_chromosome() for _ in range(pop_size)]
    best = min(population, key=lambda c: evaluate_chromosome(c)[0])
    
    for gen in range(generations):
        new_population = []
        for _ in range(pop_size):
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=lambda c: evaluate_chromosome(c)[0])
        if evaluate_chromosome(current_best)[0] < evaluate_chromosome(best)[0]:
            best = current_best
    
    return best, evaluate_chromosome(best)

# Exemple d'ús
best_chrom, (best_makespan, schedule) = genetic_algorithm()

schedule.sort(key=lambda entry: (entry[2], entry[3]))
for entry in schedule:
    print(f"Machine {entry[2]}, Job {entry[0]}, Op {entry[1]}: {entry[3]} -> {entry[4]}")
    # print(f"Job {entry[0]} - Op {entry[1]} - Machine {entry[2]}: {entry[3]} -> {entry[4]}")

print("Best schedule:", schedule)
print("Best makespan:", best_makespan)
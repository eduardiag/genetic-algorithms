import random
from matplotlib import pyplot, patches
from collections import Counter


N_EXP = 1
GENS = 1000
POP_SIZE = 100
MUTATION_RATE = 0.1
ELITE_SIZE = POP_SIZE // 20

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

    return -max(op[4] for op in schedule), schedule

# Mostra un diagrama de Gantt del schedule
def plot_gantt(schedule):
    """
    Mostra un diagrama de Gantt a partir d'un schedule:
    Cada operació es representa com una barra horitzontal segons la seva màquina i interval de temps.
    """
    fig, ax = pyplot.subplots(figsize=(12, 6))
    colors = {}

    # Assigna un color diferent per job
    job_ids = sorted(set(op[0] for op in schedule))
    colormap = pyplot.cm.get_cmap('tab20', len(job_ids))
    for i, job_id in enumerate(job_ids):
        colors[job_id] = colormap(i)

    # Dibuixa cada operació al diagrama
    for job_id, op_idx, machine_id, start, end in schedule:
        ax.barh(
            y=machine_id,
            width=end - start,
            left=start,
            height=0.8,
            color=colors[job_id],
            edgecolor='black'
        )
        ax.text(
            x=start + (end - start) / 2,
            y=machine_id,
            s=f'J{job_id}-O{op_idx}',
            va='center',
            ha='center',
            fontsize=8,
            color='white'
        )

    # Estètica
    ax.set_xlabel('Temps')
    ax.set_ylabel('Màquina')
    ax.set_title('Diagrama de Gantt del Schedule')
    ax.set_yticks(sorted(set(op[2] for op in schedule)))
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Llegenda
    legend_patches = [patches.Patch(color=colors[j], label=f'Job {j}') for j in job_ids]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    pyplot.tight_layout()
    pyplot.show()


# Selecció: torneig binari
def selection(population, k):
    selected = random.sample(population, k)
    best = second_best = None
    best_fitness = second_fitness = float('-inf')
    for individual in selected:
        f = evaluate_chromosome(individual)[0]
        if f > best_fitness:
            second_best, second_fitness = best, best_fitness
            best, best_fitness = individual, f
        elif f > second_fitness:
            second_best, second_fitness = individual, f
    return [best, second_best]

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
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(num_operations), 2))
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# Algorisme genètic
def genetic_algorithm():
    population = [generate_chromosome() for _ in range(POP_SIZE)]
    best = min(population, key=lambda c: evaluate_chromosome(c)[0])
    
    fit = []
    for gen in range(GENS):
        new_population = sorted(population, key=lambda c: evaluate_chromosome(c)[0], reverse=True)[:ELITE_SIZE]
        for _ in range(POP_SIZE - ELITE_SIZE):
            parents = selection(population, POP_SIZE // 4)
            child = crossover(*parents)
            child = mutate(child)
            new_population.append(child)
        population = new_population
        current_best = max(population, key=lambda c: evaluate_chromosome(c)[0])
        if evaluate_chromosome(best)[0] < evaluate_chromosome(current_best)[0]:
            best = current_best
        
        fit.append(-evaluate_chromosome(current_best)[0])
    convergence.append(fit)

    fitness, schedule = evaluate_chromosome(best)
    return best, -fitness, schedule





job = [(0, 0, 9, 0, 2), (15, 0, 9, 2, 11), (16, 0, 6, 0, 4), (12, 0, 5, 0, 8), (7, 0, 6, 4, 10), (19, 0, 2, 0, 7), (2, 0, 4, 0, 10), (3, 0, 9, 11, 16), (2, 1, 5, 10, 16), (17, 0, 3, 0, 7), (14, 0, 6, 10, 17), (4, 0, 1, 0, 4), (12, 1, 1, 8, 10), (7, 1, 6, 17, 25), (4, 1, 4, 10, 15), (8, 0, 8, 0, 9), (11, 0, 8, 9, 13), (13, 0, 3, 7, 15), (3, 1, 6, 25, 28), (15, 1, 3, 15, 17), (6, 0, 8, 13, 23), (12, 2, 8, 23, 24), (16, 1, 7, 4, 13), (2, 2, 1, 16, 19), (5, 0, 6, 28, 36), (15, 2, 5, 17, 27), (16, 2, 0, 13, 23), (19, 1, 4, 15, 17), (17, 1, 3, 17, 26), (18, 0, 7, 13, 16), (18, 1, 9, 16, 21), (9, 0, 6, 36, 37), (1, 0, 4, 17, 22), (0, 1, 7, 16, 18), (4, 2, 7, 18, 22), (2, 3, 3, 26, 27), (17, 2, 9, 26, 31), (4, 3, 3, 27, 35), (2, 4, 9, 31, 39), (11, 1, 8, 24, 31), (6, 1, 1, 23, 33), (12, 3, 8, 31, 35), (8, 1, 2, 9, 19), (8, 2, 4, 22, 29), (6, 2, 7, 33, 43), (16, 3, 5, 27, 31), (13, 1, 8, 35, 44), (11, 2, 5, 31, 40), (5, 1, 9, 39, 43), (9, 1, 5, 40, 45), (14, 1, 0, 23, 31), (0, 2, 2, 19, 25), (8, 3, 6, 37, 40), (5, 2, 1, 43, 48), (13, 2, 5, 45, 51), (11, 3, 7, 43, 48), (10, 0, 7, 48, 56), (0, 3, 8, 44, 46), (7, 2, 2, 25, 34), (3, 2, 3, 35, 43), (19, 2, 9, 43, 44), (17, 3, 2, 34, 37), (18, 2, 0, 31, 37), (9, 2, 3, 45, 47), (1, 1, 8, 46, 52), (17, 4, 2, 37, 47), (4, 4, 2, 47, 48), (6, 3, 9, 44, 48), (12, 4, 4, 35, 41), (16, 4, 9, 48, 50), (5, 3, 4, 48, 58), (19, 3, 9, 50, 55), (18, 3, 5, 51, 53), (9, 3, 1, 48, 56), (1, 2, 5, 53, 58), (11, 4, 2, 48, 52), (6, 4, 0, 48, 56), (0, 4, 1, 56, 65), (14, 2, 9, 55, 61), (13, 3, 4, 58, 68), (7, 3, 4, 68, 69), (3, 3, 2, 52, 61), (7, 4, 4, 69, 73), (15, 3, 2, 61, 64), (14, 3, 1, 65, 68), (18, 4, 2, 64, 72), (10, 1, 3, 56, 63), (15, 4, 4, 73, 80), (19, 4, 7, 56, 58), (9, 4, 0, 56, 63), (1, 3, 5, 58, 61), (3, 4, 0, 63, 65), (5, 4, 5, 61, 70), (8, 4, 3, 63, 64), (13, 4, 4, 80, 81), (14, 4, 1, 68, 74), (1, 4, 2, 72, 82), (10, 2, 3, 64, 65), (10, 3, 0, 65, 75), (10, 4, 7, 75, 76)]
num_jobs = len(jobs)
num_machines = max(max(op[0] for op in job) for job in jobs) + 1
job_counts = [len(job) for job in jobs]
num_operations = sum(job_counts)

chrom, makespan, schedule = genetic_algorithm()
    
print("Jobs:", jobs)

# Visualització del millor schedule
plot_gantt(top_schedule)
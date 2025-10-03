import random
from matplotlib import pyplot, patches
from collections import Counter


N_EXP = 1
GENS = 200
POP_SIZE = 30
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

    return -max(job_available), schedule

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

# Visualització de la convergència
def plot_convergence(convergence):
    pyplot.figure(figsize=(10, 6))
    for fit in convergence:
        pyplot.plot(fit)
    pyplot.title("Convergence curves")
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Best fitness")
    pyplot.grid(True)
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

def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    left = Counter(parent1) - Counter(child[a:b])
    j = 0
    for i in range(size):
        if child[i] is None:
            while True:
                job = parent2[j % size]
                j += 1
                if left[job]:
                    child[i] = job
                    left[job] -= 1
                    break
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
        print("Generation", gen, "Best Fitness:", fit[-1])
    convergence.append(fit)

    fitness, schedule = evaluate_chromosome(best)
    return best, -fitness, schedule



# jobs = generate_random_jobs(60, 30, 30, 1, 20)
# jobs = [[(0, 3), (1, 2), (2, 2)], [(0, 2), (2, 1), (1, 4)], [(1, 4), (2, 3), (0, 2)]]
# jobs = [[(9, 2), (7, 2), (2, 6), (8, 2), (1, 9)], [(4, 5), (8, 6), (5, 5), (5, 3), (2, 10)], [(4, 10), (5, 6), (1, 3), (3, 1), (9, 8)], [(9, 5), (6, 3), (3, 8), (2, 9), (0, 2)], [(1, 4), (4, 5), (7, 4), (3, 8), (2, 1)], [(6, 8), (9, 4), (1, 5), (4, 10), (5, 9)], [(8, 10), (1, 10), (7, 10), (9, 4), (0, 8)], [(6, 6), (6, 8), (2, 9), (4, 1), (4, 4)], [(8, 9), (2, 10), (4, 7), (6, 3), (3, 1)], [(6, 1), (5, 5), (3, 2), (1, 8), (0, 7)], [(7, 8), (3, 7), (3, 1), (0, 10), (7, 1)], [(8, 4), (8, 7), (5, 9), (7, 5), (2, 4)], [(5, 8), (1, 2), (8, 1), (8, 4), (4, 6)], [(3, 8), (8, 9), (5, 6), (4, 10), (4, 1)], [(6, 7), (0, 8), (9, 6), (1, 3), (1, 6)], [(9, 9), (3, 2), (5, 10), (2, 3), (4, 7)], [(6, 4), (7, 9), (0, 10), (5, 4), (9, 2)], [(3, 7), (3, 9), (9, 5), (2, 3), (2, 10)], [(7, 3), (9, 5), (0, 6), (5, 2), (2, 8)], [(2, 7), (4, 2), (9, 1), (9, 5), (7, 2)]]
# jobs = [[(5, 2), (3, 2), (8, 5), (2, 1), (3, 5)], [(2, 4), (8, 2), (8, 1), (2, 1), (9, 1)], [(6, 3), (1, 2), (5, 2), (5, 5), (9, 4)], [(6, 3), (5, 2), (4, 1), (8, 4), (1, 1)], [(9, 5), (2, 1), (6, 1), (1, 3), (2, 3)], [(0, 3), (9, 2), (2, 3), (0, 1), (2, 5)], [(3, 4), (1, 1), (2, 5), (3, 1), (3, 5)], [(3, 2), (5, 4), (6, 2), (5, 1), (2, 2)], [(4, 2), (2, 3), (1, 1), (4, 3), (2, 2)], [(4, 3), (9, 1), (3, 2), (4, 5), (7, 1)], [(3, 3), (9, 3), (0, 5), (5, 1), (0, 5)], [(9, 1), (6, 1), (7, 4), (2, 3), (1, 1)], [(7, 1), (2, 2), (6, 2), (9, 5), (9, 5)], [(1, 2), (4, 1), (4, 3), (4, 3), (4, 4)], [(0, 1), (0, 2), (9, 2), (3, 2), (5, 3)], [(1, 2), (0, 2), (7, 3), (3, 3), (2, 3)], [(2, 2), (0, 1), (8, 3), (4, 1), (7, 5)], [(7, 3), (7, 1), (2, 4), (7, 3), (0, 5)], [(6, 4), (8, 4), (4, 4), (0, 2), (9, 3)], [(3, 4), (1, 4), (8, 3), (6, 3), (6, 2)], [(3, 4), (5, 1), (5, 5), (5, 3), (4, 5)], [(8, 4), (5, 3), (6, 1), (1, 3), (6, 3)], [(3, 4), (2, 2), (2, 5), (2, 1), (6, 5)], [(0, 4), (4, 1), (1, 5), (8, 1), (3, 4)], [(4, 3), (4, 1), (2, 1), (6, 3), (9, 4)], [(1, 1), (9, 1), (9, 4), (5, 4), (7, 3)], [(6, 2), (3, 3), (7, 2), (4, 5), (1, 4)], [(9, 4), (9, 5), (3, 1), (4, 4), (4, 4)], [(7, 1), (1, 2), (1, 4), (8, 1), (2, 1)], [(6, 1), (5, 2), (6, 5), (0, 2), (6, 5)], [(9, 4), (7, 3), (0, 1), (9, 3), (1, 3)], [(0, 3), (8, 3), (7, 5), (0, 4), (8, 1)], [(7, 5), (9, 1), (2, 3), (1, 1), (6, 5)], [(4, 4), (6, 3), (7, 1), (4, 5), (2, 4)], [(6, 5), (7, 4), (0, 4), (5, 5), (3, 3)], [(8, 2), (5, 3), (9, 2), (9, 4), (2, 4)], [(0, 3), (9, 4), (9, 2), (8, 2), (4, 4)], [(0, 3), (0, 2), (4, 4), (1, 5), (7, 5)], [(2, 2), (4, 1), (2, 1), (8, 5), (1, 3)], [(6, 1), (8, 2), (1, 2), (3, 4), (5, 3)], [(0, 5), (9, 5), (7, 3), (1, 3), (8, 5)], [(1, 2), (2, 5), (5, 3), (9, 4), (7, 2)], [(8, 1), (8, 1), (5, 1), (1, 4), (9, 5)], [(4, 1), (8, 5), (5, 1), (7, 4), (8, 5)], [(7, 2), (5, 1), (0, 1), (6, 3), (4, 3)], [(5, 4), (8, 2), (5, 2), (4, 3), (4, 4)], [(2, 4), (5, 5), (7, 3), (8, 2), (8, 5)], [(0, 1), (4, 2), (3, 4), (3, 5), (2, 3)], [(4, 4), (3, 1), (5, 1), (2, 1), (3, 4)], [(3, 1), (2, 5), (4, 5), (0, 2), (1, 1)], [(1, 5), (4, 2), (1, 2), (2, 4), (6, 4)], [(2, 1), (8, 1), (0, 4), (3, 4), (0, 3)], [(7, 3), (1, 4), (1, 2), (8, 4), (9, 4)], [(0, 1), (2, 5), (4, 5), (2, 4), (1, 1)], [(7, 4), (9, 3), (1, 1), (1, 1), (3, 3)], [(5, 3), (8, 3), (6, 3), (4, 1), (1, 5)], [(9, 2), (5, 5), (1, 5), (6, 2), (3, 2)], [(5, 3), (4, 3), (5, 4), (1, 3), (2, 5)], [(8, 3), (5, 1), (2, 4), (9, 2), (6, 3)], [(4, 1), (5, 4), (6, 2), (0, 4), (0, 4)]]
# jobs = [[(1, 1), (19, 2), (6, 1), (1, 6), (9, 2), (9, 9), (6, 1), (18, 2), (5, 3), (18, 2), (5, 7), (1, 2), (10, 8), (17, 4), (5, 8), (7, 2), (1, 5), (2, 3), (6, 2), (12, 8)], [(3, 3), (11, 2), (2, 1), (7, 2), (2, 2), (15, 5), (9, 1), (0, 4), (5, 8), (14, 2), (8, 2), (6, 5), (16, 5), (0, 8), (14, 2), (10, 3), (2, 10), (13, 5), (13, 9), (8, 4)], [(13, 9), (19, 3), (13, 7), (8, 10), (12, 9), (10, 3), (18, 5), (0, 2), (1, 10), (3, 2), (3, 3), (16, 3), (19, 7), (7, 7), (5, 1), (9, 7), (5, 9), (7, 5), (3, 9), (5, 7)], [(12, 2), (11, 7), (15, 4), (10, 3), (3, 6), (14, 8), (13, 3), (0, 8), (14, 4), (14, 4), (0, 8), (4, 7), (9, 8), (15, 9), (8, 8), (8, 9), (8, 10), (13, 8), (10, 3), (16, 9)], [(2, 5), (0, 6), (17, 3), (5, 2), (18, 10), (16, 9), (18, 5), (5, 2), (14, 6), (8, 6), (8, 4), (5, 2), (18, 10), (8, 5), (10, 10), (18, 2), (8, 2), (19, 5), (7, 5), (16, 1)], [(14, 10), (13, 6), (14, 3), (3, 8), (19, 7), (12, 2), (13, 7), (8, 4), (18, 5), (5, 5), (14, 4), (18, 7), (9, 8), (7, 10), (15, 6), (18, 3), (18, 9), (11, 5), (12, 3), (1, 4)], [(6, 8), (1, 9), (4, 3), (7, 10), (10, 9), (10, 7), (14, 2), (15, 4), (4, 7), (1, 10), (19, 1), (13, 5), (18, 10), (12, 7), (19, 8), (2, 8), (16, 6), (15, 2), (12, 9), (5, 4)], [(8, 6), (5, 7), (18, 8), (10, 3), (5, 4), (19, 2), (14, 3), (13, 10), (17, 8), (14, 3), (13, 9), (15, 4), (7, 5), (13, 6), (10, 6), (0, 2), (14, 10), (3, 1), (3, 2), (19, 8)], [(12, 7), (17, 3), (3, 3), (1, 3), (4, 2), (18, 7), (12, 8), (17, 8), (0, 7), (15, 2), (2, 6), (0, 5), (12, 7), (3, 3), (3, 7), (17, 5), (0, 8), (3, 10), (12, 8), (13, 7)], [(17, 8), (1, 4), (19, 3), (9, 5), (7, 6), (2, 9), (19, 6), (16, 4), (17, 8), (14, 10), (10, 9), (7, 3), (4, 1), (16, 5), (12, 4), (0, 9), (14, 4), (19, 2), (5, 5), (8, 1)], [(2, 3), (3, 10), (9, 8), (18, 5), (16, 2), (18, 2), (19, 9), (0, 8), (18, 7), (17, 1), (13, 7), (15, 10), (14, 3), (10, 10), (2, 5), (17, 3), (8, 8), (3, 5), (4, 9), (0, 10)], [(13, 3), (12, 6), (15, 3), (19, 2), (10, 10), (9, 7), (15, 10), (10, 5), (12, 3), (1, 9), (1, 9), (12, 2), (10, 1), (16, 6), (16, 10), (9, 7), (12, 9), (1, 8), (4, 9), (7, 2)], [(14, 4), (8, 4), (16, 4), (12, 3), (0, 9), (2, 3), (9, 7), (8, 1), (11, 6), (10, 8), (6, 1), (11, 10), (9, 3), (13, 2), (17, 1), (13, 4), (15, 9), (14, 10), (0, 4), (4, 5)], [(9, 6), (16, 3), (7, 5), (1, 9), (14, 8), (17, 2), (5, 10), (11, 10), (11, 1), (3, 7), (7, 10), (14, 4), (15, 8), (14, 6), (19, 10), (18, 5), (0, 7), (8, 2), (4, 3), (1, 10)], [(3, 5), (3, 1), (4, 8), (10, 3), (17, 5), (14, 8), (16, 4), (12, 5), (15, 6), (4, 8), (19, 9), (15, 2), (18, 8), (0, 1), (0, 6), (18, 6), (19, 10), (15, 1), (3, 1), (8, 9)], [(9, 5), (14, 5), (5, 2), (2, 10), (15, 8), (18, 9), (15, 7), (9, 9), (9, 6), (14, 8), (8, 3), (4, 4), (13, 6), (16, 8), (15, 3), (4, 2), (15, 5), (3, 4), (18, 4), (15, 6)], [(19, 1), (13, 10), (8, 10), (10, 4), (6, 2), (18, 5), (13, 2), (16, 9), (8, 5), (16, 4), (10, 6), (3, 3), (11, 5), (1, 3), (13, 8), (12, 10), (16, 6), (9, 1), (16, 6), (15, 6)], [(3, 5), (18, 10), (15, 4), (15, 5), (16, 3), (12, 9), (10, 7), (17, 7), (0, 9), (3, 7), (8, 5), (2, 2), (0, 7), (1, 5), (17, 10), (17, 1), (13, 4), (16, 7), (8, 2), (2, 1)], [(7, 5), (3, 3), (2, 1), (12, 7), (18, 9), (16, 9), (17, 3), (0, 2), (9, 5), (8, 4), (10, 10), (18, 7), (4, 2), (3, 1), (17, 4), (12, 6), (2, 3), (10, 4), (19, 9), (17, 8)], [(6, 6), (5, 10), (16, 1), (7, 1), (12, 4), (4, 5), (18, 1), (3, 1), (1, 8), (0, 5), (18, 1), (17, 4), (6, 4), (18, 1), (1, 1), (1, 6), (16, 6), (14, 6), (7, 6), (14, 5)]]
jobs = generate_random_jobs(25, 25, 25, 1, 10)

num_jobs = len(jobs)
num_machines = max(max(op[0] for op in job) for job in jobs) + 1
job_counts = [len(job) for job in jobs]
num_operations = sum(job_counts)

convergence = []
top_chromosome, top_schedule, top_makespan = [], [], float('inf')
# Exemple d'ús
for exp in range(N_EXP):
    print("Running experiment", exp, "...")  
    best_chrom, best_makespan, best_schedule = genetic_algorithm()
    if best_makespan < top_makespan:
        top_chromosome = best_chrom
        top_schedule = best_schedule
        top_makespan = best_makespan
    
    """
    best_schedule.sort(key=lambda entry: (entry[2], entry[3]))
    for entry in best_schedule:
        print(f"Machine {entry[2]}, Job {entry[0]}, Op {entry[1]}: {entry[3]} -> {entry[4]}")
        print(f"Job {entry[0]} - Op {entry[1]} - Machine {entry[2]}: {entry[3]} -> {entry[4]}")
    """

print("Jobs:", jobs)
print("Best Overall Schedule:", top_schedule)
print("Best Overall Makespan:", top_makespan)
print("Parameters:", GENS, POP_SIZE, MUTATION_RATE)



# Visualització de la convergència
plot_convergence(convergence)

# Visualització del millor schedule
plot_gantt(top_schedule)
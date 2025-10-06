import time
import random
from matplotlib import pyplot, patches
from collections import Counter


N_EXP = 1
GENS = 500
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



num_jobs = 7
num_machines = 2
num_operations = 100
min_duration = 1
max_duration = 10

jobs = generate_random_jobs(num_jobs, num_machines, num_operations, min_duration, max_duration)
job_counts = [len(job) for job in jobs]
total_operations = sum(job_counts)

convergence = []
top_chromosome, top_schedule, top_makespan = [], [], float('inf')


for exp in range(N_EXP):
    print("Running experiment", exp, "...")

    start_time = time.time()

    best_chrom, best_makespan, best_schedule = genetic_algorithm()
    if best_makespan < top_makespan:
        top_chromosome = best_chrom
        top_schedule = best_schedule
        top_makespan = best_makespan
    
    total_time = time.time() - start_time    
    fitxer = r"C:\Users\EduardGonzalvoGelabe\OneDrive - Capitole Consulting\Documentos\codis\genetic-algorithms\job_scheduling\results.txt"
    with open(fitxer, "a", encoding="utf-8") as f:
        f.write(f"GA Manual\t{num_jobs}\t{num_machines}\t{num_operations}\t{int(best_makespan)}\t{total_time:.2f}\t{GENS}\t{POP_SIZE}\t{MUTATION_RATE}\n")

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
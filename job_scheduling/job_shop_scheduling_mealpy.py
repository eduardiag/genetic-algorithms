from mealpy import FloatVar, Problem, GA, PSO
from matplotlib import pyplot, patches
import random

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

# Dades generades aleatòriament
# jobs = generate_random_jobs(num_jobs=60, num_machines=40, operations_per_job=30, min_duration=1, max_duration=10)
jobs = [[(5, 2), (3, 2), (8, 5), (2, 1), (3, 5)], [(2, 4), (8, 2), (8, 1), (2, 1), (9, 1)], [(6, 3), (1, 2), (5, 2), (5, 5), (9, 4)], [(6, 3), (5, 2), (4, 1), (8, 4), (1, 1)], [(9, 5), (2, 1), (6, 1), (1, 3), (2, 3)], [(0, 3), (9, 2), (2, 3), (0, 1), (2, 5)], [(3, 4), (1, 1), (2, 5), (3, 1), (3, 5)], [(3, 2), (5, 4), (6, 2), (5, 1), (2, 2)], [(4, 2), (2, 3), (1, 1), (4, 3), (2, 2)], [(4, 3), (9, 1), (3, 2), (4, 5), (7, 1)], [(3, 3), (9, 3), (0, 5), (5, 1), (0, 5)], [(9, 1), (6, 1), (7, 4), (2, 3), (1, 1)], [(7, 1), (2, 2), (6, 2), (9, 5), (9, 5)], [(1, 2), (4, 1), (4, 3), (4, 3), (4, 4)], [(0, 1), (0, 2), (9, 2), (3, 2), (5, 3)], [(1, 2), (0, 2), (7, 3), (3, 3), (2, 3)], [(2, 2), (0, 1), (8, 3), (4, 1), (7, 5)], [(7, 3), (7, 1), (2, 4), (7, 3), (0, 5)], [(6, 4), (8, 4), (4, 4), (0, 2), (9, 3)], [(3, 4), (1, 4), (8, 3), (6, 3), (6, 2)], [(3, 4), (5, 1), (5, 5), (5, 3), (4, 5)], [(8, 4), (5, 3), (6, 1), (1, 3), (6, 3)], [(3, 4), (2, 2), (2, 5), (2, 1), (6, 5)], [(0, 4), (4, 1), (1, 5), (8, 1), (3, 4)], [(4, 3), (4, 1), (2, 1), (6, 3), (9, 4)], [(1, 1), (9, 1), (9, 4), (5, 4), (7, 3)], [(6, 2), (3, 3), (7, 2), (4, 5), (1, 4)], [(9, 4), (9, 5), (3, 1), (4, 4), (4, 4)], [(7, 1), (1, 2), (1, 4), (8, 1), (2, 1)], [(6, 1), (5, 2), (6, 5), (0, 2), (6, 5)], [(9, 4), (7, 3), (0, 1), (9, 3), (1, 3)], [(0, 3), (8, 3), (7, 5), (0, 4), (8, 1)], [(7, 5), (9, 1), (2, 3), (1, 1), (6, 5)], [(4, 4), (6, 3), (7, 1), (4, 5), (2, 4)], [(6, 5), (7, 4), (0, 4), (5, 5), (3, 3)], [(8, 2), (5, 3), (9, 2), (9, 4), (2, 4)], [(0, 3), (9, 4), (9, 2), (8, 2), (4, 4)], [(0, 3), (0, 2), (4, 4), (1, 5), (7, 5)], [(2, 2), (4, 1), (2, 1), (8, 5), (1, 3)], [(6, 1), (8, 2), (1, 2), (3, 4), (5, 3)], [(0, 5), (9, 5), (7, 3), (1, 3), (8, 5)], [(1, 2), (2, 5), (5, 3), (9, 4), (7, 2)], [(8, 1), (8, 1), (5, 1), (1, 4), (9, 5)], [(4, 1), (8, 5), (5, 1), (7, 4), (8, 5)], [(7, 2), (5, 1), (0, 1), (6, 3), (4, 3)], [(5, 4), (8, 2), (5, 2), (4, 3), (4, 4)], [(2, 4), (5, 5), (7, 3), (8, 2), (8, 5)], [(0, 1), (4, 2), (3, 4), (3, 5), (2, 3)], [(4, 4), (3, 1), (5, 1), (2, 1), (3, 4)], [(3, 1), (2, 5), (4, 5), (0, 2), (1, 1)], [(1, 5), (4, 2), (1, 2), (2, 4), (6, 4)], [(2, 1), (8, 1), (0, 4), (3, 4), (0, 3)], [(7, 3), (1, 4), (1, 2), (8, 4), (9, 4)], [(0, 1), (2, 5), (4, 5), (2, 4), (1, 1)], [(7, 4), (9, 3), (1, 1), (1, 1), (3, 3)], [(5, 3), (8, 3), (6, 3), (4, 1), (1, 5)], [(9, 2), (5, 5), (1, 5), (6, 2), (3, 2)], [(5, 3), (4, 3), (5, 4), (1, 3), (2, 5)], [(8, 3), (5, 1), (2, 4), (9, 2), (6, 3)], [(4, 1), (5, 4), (6, 2), (0, 4), (0, 4)]]
jobs = [[(1, 1), (19, 2), (6, 1), (1, 6), (9, 2), (9, 9), (6, 1), (18, 2), (5, 3), (18, 2), (5, 7), (1, 2), (10, 8), (17, 4), (5, 8), (7, 2), (1, 5), (2, 3), (6, 2), (12, 8)], [(3, 3), (11, 2), (2, 1), (7, 2), (2, 2), (15, 5), (9, 1), (0, 4), (5, 8), (14, 2), (8, 2), (6, 5), (16, 5), (0, 8), (14, 2), (10, 3), (2, 10), (13, 5), (13, 9), (8, 4)], [(13, 9), (19, 3), (13, 7), (8, 10), (12, 9), (10, 3), (18, 5), (0, 2), (1, 10), (3, 2), (3, 3), (16, 3), (19, 7), (7, 7), (5, 1), (9, 7), (5, 9), (7, 5), (3, 9), (5, 7)], [(12, 2), (11, 7), (15, 4), (10, 3), (3, 6), (14, 8), (13, 3), (0, 8), (14, 4), (14, 4), (0, 8), (4, 7), (9, 8), (15, 9), (8, 8), (8, 9), (8, 10), (13, 8), (10, 3), (16, 9)], [(2, 5), (0, 6), (17, 3), (5, 2), (18, 10), (16, 9), (18, 5), (5, 2), (14, 6), (8, 6), (8, 4), (5, 2), (18, 10), (8, 5), (10, 10), (18, 2), (8, 2), (19, 5), (7, 5), (16, 1)], [(14, 10), (13, 6), (14, 3), (3, 8), (19, 7), (12, 2), (13, 7), (8, 4), (18, 5), (5, 5), (14, 4), (18, 7), (9, 8), (7, 10), (15, 6), (18, 3), (18, 9), (11, 5), (12, 3), (1, 4)], [(6, 8), (1, 9), (4, 3), (7, 10), (10, 9), (10, 7), (14, 2), (15, 4), (4, 7), (1, 10), (19, 1), (13, 5), (18, 10), (12, 7), (19, 8), (2, 8), (16, 6), (15, 2), (12, 9), (5, 4)], [(8, 6), (5, 7), (18, 8), (10, 3), (5, 4), (19, 2), (14, 3), (13, 10), (17, 8), (14, 3), (13, 9), (15, 4), (7, 5), (13, 6), (10, 6), (0, 2), (14, 10), (3, 1), (3, 2), (19, 8)], [(12, 7), (17, 3), (3, 3), (1, 3), (4, 2), (18, 7), (12, 8), (17, 8), (0, 7), (15, 2), (2, 6), (0, 5), (12, 7), (3, 3), (3, 7), (17, 5), (0, 8), (3, 10), (12, 8), (13, 7)], [(17, 8), (1, 4), (19, 3), (9, 5), (7, 6), (2, 9), (19, 6), (16, 4), (17, 8), (14, 10), (10, 9), (7, 3), (4, 1), (16, 5), (12, 4), (0, 9), (14, 4), (19, 2), (5, 5), (8, 1)], [(2, 3), (3, 10), (9, 8), (18, 5), (16, 2), (18, 2), (19, 9), (0, 8), (18, 7), (17, 1), (13, 7), (15, 10), (14, 3), (10, 10), (2, 5), (17, 3), (8, 8), (3, 5), (4, 9), (0, 10)], [(13, 3), (12, 6), (15, 3), (19, 2), (10, 10), (9, 7), (15, 10), (10, 5), (12, 3), (1, 9), (1, 9), (12, 2), (10, 1), (16, 6), (16, 10), (9, 7), (12, 9), (1, 8), (4, 9), (7, 2)], [(14, 4), (8, 4), (16, 4), (12, 3), (0, 9), (2, 3), (9, 7), (8, 1), (11, 6), (10, 8), (6, 1), (11, 10), (9, 3), (13, 2), (17, 1), (13, 4), (15, 9), (14, 10), (0, 4), (4, 5)], [(9, 6), (16, 3), (7, 5), (1, 9), (14, 8), (17, 2), (5, 10), (11, 10), (11, 1), (3, 7), (7, 10), (14, 4), (15, 8), (14, 6), (19, 10), (18, 5), (0, 7), (8, 2), (4, 3), (1, 10)], [(3, 5), (3, 1), (4, 8), (10, 3), (17, 5), (14, 8), (16, 4), (12, 5), (15, 6), (4, 8), (19, 9), (15, 2), (18, 8), (0, 1), (0, 6), (18, 6), (19, 10), (15, 1), (3, 1), (8, 9)], [(9, 5), (14, 5), (5, 2), (2, 10), (15, 8), (18, 9), (15, 7), (9, 9), (9, 6), (14, 8), (8, 3), (4, 4), (13, 6), (16, 8), (15, 3), (4, 2), (15, 5), (3, 4), (18, 4), (15, 6)], [(19, 1), (13, 10), (8, 10), (10, 4), (6, 2), (18, 5), (13, 2), (16, 9), (8, 5), (16, 4), (10, 6), (3, 3), (11, 5), (1, 3), (13, 8), (12, 10), (16, 6), (9, 1), (16, 6), (15, 6)], [(3, 5), (18, 10), (15, 4), (15, 5), (16, 3), (12, 9), (10, 7), (17, 7), (0, 9), (3, 7), (8, 5), (2, 2), (0, 7), (1, 5), (17, 10), (17, 1), (13, 4), (16, 7), (8, 2), (2, 1)], [(7, 5), (3, 3), (2, 1), (12, 7), (18, 9), (16, 9), (17, 3), (0, 2), (9, 5), (8, 4), (10, 10), (18, 7), (4, 2), (3, 1), (17, 4), (12, 6), (2, 3), (10, 4), (19, 9), (17, 8)], [(6, 6), (5, 10), (16, 1), (7, 1), (12, 4), (4, 5), (18, 1), (3, 1), (1, 8), (0, 5), (18, 1), (17, 4), (6, 4), (18, 1), (1, 1), (1, 6), (16, 6), (14, 6), (7, 6), (14, 5)]]
jobs = generate_random_jobs(7, 2, 100, 1, 10)

n_jobs = len(jobs)
n_machines = len(set(m for job in jobs for m, _ in job))
n_operations = sum(len(job) for job in jobs)

def decode_schedule(seq):
    """
    Converteix una seqüència numèrica en una planificació concreta i retorna el makespan.
    seq: vector amb números reals [0, 1] -> ordenem per prioritat per cada job
    """
    # Indexació d'operacions per job
    job_ops_remaining = [list(range(len(job))) for job in jobs]
    job_times = [0] * n_jobs
    machine_times = [0] * n_machines

    # Transformem seq en una llista d'ordres per executar
    job_priorities = [[] for _ in range(n_jobs)]
    idx = 0

    for j in range(n_jobs):
        for _ in range(len(jobs[j])):
            job_priorities[j].append((seq[idx], _))
            idx += 1
        job_priorities[j].sort()

    # Simulació
    schedule = []
    while any(job_ops_remaining):
        for j in range(n_jobs):
            if job_ops_remaining[j]:
                op_idx = job_priorities[j][0][1]
                if op_idx in job_ops_remaining[j]:
                    machine, duration = jobs[j][op_idx]
                    start = max(job_times[j], machine_times[machine])
                    end = start + duration
                    job_times[j] = end
                    machine_times[machine] = end
                    job_ops_remaining[j].remove(op_idx)
                    job_priorities[j].pop(0)
                    schedule.append((j, op_idx, machine, start, end))

    return max(job_times), schedule

def fitness_function(solution):
    """Funció objectiu que retorna el makespan (menor és millor)"""
    return decode_schedule(solution)[0]

def plot_gantt(schedule):
    """
    Mostra un diagrama de Gantt a partir d'un schedule:
    Cada operació es representa com una barra horitzontal segons la seva màquina i interval de temps.
    """
    fig, ax = pyplot.subplots(figsize=(12, 6))
    colors = {}

    # Assigna un color diferent per job
    job_ids = sorted(set(op[0] for op in schedule))
    colormap = pyplot.get_cmap('tab20', len(job_ids))
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



print("Iniciant optimització Job Shop Scheduling...")
print(f"Problema: {n_jobs} jobs, {n_machines} màquines, {n_operations} operacions")

import time
start_time = time.time()

problem = Problem(
    obj_func=fitness_function,
    bounds=[FloatVar(lb=0.0, ub=1.0, name=f"priority_{i}") for i in range(n_operations)],
    minmax="min",
    name="JobShopScheduling"
)




# ------------------------------------------------------------

algorithm = "GA"
epoch = 200
pop_size = 30

if algorithm == "GA":
    model = GA.BaseGA(epoch, pop_size)
elif algorithm == "PSO":
    model = PSO.OriginalPSO(epoch, pop_size)

best_agent = model.solve(problem)
best_solution = best_agent.solution
best_fitness = best_agent.target.fitness

elapsed_time = time.time() - start_time
fitxer = r"C:\Users\EduardGonzalvoGelabe\OneDrive - Capitole Consulting\Documentos\codis\genetic-algorithms\job_scheduling\results.txt"
with open(fitxer, "a", encoding="utf-8") as f:
    f.write(f"Mealpy {algorithm}\t{n_jobs}\t{n_machines}\t{n_operations}\t{best_fitness}\t{elapsed_time:.2f}\n")




print("\n" + "="*50)
print("RESULTATS DE L'OPTIMITZACIÓ")
print("="*50)
print(f"Makespan mínim trobat: {best_fitness:.2f} unitats de temps")

print(f"\nDetalls del problema:")
print(f"- Nombre de jobs: {n_jobs}")
print(f"- Nombre de màquines: {n_machines}") 
print(f"- Nombre total d'operacions: {n_operations}")

print(best_solution)
schedule = decode_schedule(best_solution)[1]
plot_gantt(schedule)
print(jobs)
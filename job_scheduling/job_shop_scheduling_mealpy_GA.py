from mealpy import FloatVar, Problem, GA
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
jobs = generate_random_jobs(num_jobs=30, num_machines=20, operations_per_job=10, min_duration=1, max_duration=10)
jobs = [[(5, 2), (3, 2), (8, 5), (2, 1), (3, 5)], [(2, 4), (8, 2), (8, 1), (2, 1), (9, 1)], [(6, 3), (1, 2), (5, 2), (5, 5), (9, 4)], [(6, 3), (5, 2), (4, 1), (8, 4), (1, 1)], [(9, 5), (2, 1), (6, 1), (1, 3), (2, 3)], [(0, 3), (9, 2), (2, 3), (0, 1), (2, 5)], [(3, 4), (1, 1), (2, 5), (3, 1), (3, 5)], [(3, 2), (5, 4), (6, 2), (5, 1), (2, 2)], [(4, 2), (2, 3), (1, 1), (4, 3), (2, 2)], [(4, 3), (9, 1), (3, 2), (4, 5), (7, 1)], [(3, 3), (9, 3), (0, 5), (5, 1), (0, 5)], [(9, 1), (6, 1), (7, 4), (2, 3), (1, 1)], [(7, 1), (2, 2), (6, 2), (9, 5), (9, 5)], [(1, 2), (4, 1), (4, 3), (4, 3), (4, 4)], [(0, 1), (0, 2), (9, 2), (3, 2), (5, 3)], [(1, 2), (0, 2), (7, 3), (3, 3), (2, 3)], [(2, 2), (0, 1), (8, 3), (4, 1), (7, 5)], [(7, 3), (7, 1), (2, 4), (7, 3), (0, 5)], [(6, 4), (8, 4), (4, 4), (0, 2), (9, 3)], [(3, 4), (1, 4), (8, 3), (6, 3), (6, 2)], [(3, 4), (5, 1), (5, 5), (5, 3), (4, 5)], [(8, 4), (5, 3), (6, 1), (1, 3), (6, 3)], [(3, 4), (2, 2), (2, 5), (2, 1), (6, 5)], [(0, 4), (4, 1), (1, 5), (8, 1), (3, 4)], [(4, 3), (4, 1), (2, 1), (6, 3), (9, 4)], [(1, 1), (9, 1), (9, 4), (5, 4), (7, 3)], [(6, 2), (3, 3), (7, 2), (4, 5), (1, 4)], [(9, 4), (9, 5), (3, 1), (4, 4), (4, 4)], [(7, 1), (1, 2), (1, 4), (8, 1), (2, 1)], [(6, 1), (5, 2), (6, 5), (0, 2), (6, 5)], [(9, 4), (7, 3), (0, 1), (9, 3), (1, 3)], [(0, 3), (8, 3), (7, 5), (0, 4), (8, 1)], [(7, 5), (9, 1), (2, 3), (1, 1), (6, 5)], [(4, 4), (6, 3), (7, 1), (4, 5), (2, 4)], [(6, 5), (7, 4), (0, 4), (5, 5), (3, 3)], [(8, 2), (5, 3), (9, 2), (9, 4), (2, 4)], [(0, 3), (9, 4), (9, 2), (8, 2), (4, 4)], [(0, 3), (0, 2), (4, 4), (1, 5), (7, 5)], [(2, 2), (4, 1), (2, 1), (8, 5), (1, 3)], [(6, 1), (8, 2), (1, 2), (3, 4), (5, 3)], [(0, 5), (9, 5), (7, 3), (1, 3), (8, 5)], [(1, 2), (2, 5), (5, 3), (9, 4), (7, 2)], [(8, 1), (8, 1), (5, 1), (1, 4), (9, 5)], [(4, 1), (8, 5), (5, 1), (7, 4), (8, 5)], [(7, 2), (5, 1), (0, 1), (6, 3), (4, 3)], [(5, 4), (8, 2), (5, 2), (4, 3), (4, 4)], [(2, 4), (5, 5), (7, 3), (8, 2), (8, 5)], [(0, 1), (4, 2), (3, 4), (3, 5), (2, 3)], [(4, 4), (3, 1), (5, 1), (2, 1), (3, 4)], [(3, 1), (2, 5), (4, 5), (0, 2), (1, 1)], [(1, 5), (4, 2), (1, 2), (2, 4), (6, 4)], [(2, 1), (8, 1), (0, 4), (3, 4), (0, 3)], [(7, 3), (1, 4), (1, 2), (8, 4), (9, 4)], [(0, 1), (2, 5), (4, 5), (2, 4), (1, 1)], [(7, 4), (9, 3), (1, 1), (1, 1), (3, 3)], [(5, 3), (8, 3), (6, 3), (4, 1), (1, 5)], [(9, 2), (5, 5), (1, 5), (6, 2), (3, 2)], [(5, 3), (4, 3), (5, 4), (1, 3), (2, 5)], [(8, 3), (5, 1), (2, 4), (9, 2), (6, 3)], [(4, 1), (5, 4), (6, 2), (0, 4), (0, 4)]]

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

problem = Problem(
    obj_func=fitness_function,
    bounds=[FloatVar(lb=0.0, ub=1.0, name=f"priority_{i}") for i in range(n_operations)],
    minmax="min",
    name="JobShopScheduling"
)

model = GA.BaseGA(epoch=200, pop_size=30)

print("Iniciant optimització Job Shop Scheduling...")
print(f"Problema: {n_jobs} jobs, {n_machines} màquines, {n_operations} operacions")

best_agent = model.solve(problem)
best_solution = best_agent.solution
best_fitness = best_agent.target.fitness

print("\n" + "="*50)
print("RESULTATS DE L'OPTIMITZACIÓ")
print("="*50)
print(f"Makespan mínim trobat: {best_fitness:.2f} unitats de temps")

print(f"\nDetalls del problema:")
print(f"- Nombre de jobs: {n_jobs}")
print(f"- Nombre de màquines: {n_machines}") 
print(f"- Nombre total d'operacions: {n_operations}")








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



print(best_solution)
schedule = decode_schedule(best_solution)[1]
plot_gantt(schedule)
from mealpy import FloatVar, PSO, Problem
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
jobs = generate_random_jobs(num_jobs=20, num_machines=40, operations_per_job=30, min_duration=1, max_duration=10)

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

model = PSO.OriginalPSO(epoch=200, pop_size=30)

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



print(best_solution)
schedule = decode_schedule(best_solution)[1]
plot_gantt(schedule)
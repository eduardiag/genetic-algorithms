from matplotlib import pyplot, patches
from ortools.sat.python import cp_model
from random import randint
import time


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
            machine_id = randint(0, num_machines - 1)
            duration = randint(min_duration, max_duration)
            job.append((machine_id, duration))
        jobs.append(job)
    
    return jobs


def plot_gantt(schedule):
    fig, ax = pyplot.subplots(figsize=(12, 6))
    colors = {}
    job_ids = sorted(set(op[0] for op in schedule))
    colormap = pyplot.cm.get_cmap('tab20', len(job_ids))
    for i, job_id in enumerate(job_ids):
        colors[job_id] = colormap(i)

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

    ax.set_xlabel('Temps')
    ax.set_ylabel('Màquina')
    ax.set_title('Diagrama de Gantt del Schedule')
    ax.set_yticks(sorted(set(op[2] for op in schedule)))
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    legend_patches = [patches.Patch(color=colors[j], label=f'Job {j}') for j in job_ids]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    pyplot.tight_layout()
    pyplot.show()


def cp_solver(jobs, time_limit_seconds=10):
    model = cp_model.CpModel()
    num_machines = max(max(op[0] for op in job) for job in jobs) + 1

    # Càlcul d'un horizon més realista
    horizon = sum(duration for job in jobs for _, duration in job)
    print(f"Horizon calculat: {horizon}")
    
    all_ops = []
    intervals_per_machine = [[] for _ in range(num_machines)]
    starts = []
    ends = []
    
    print(f"Creant model per {len(jobs)} jobs i {num_machines} màquines...")

    for job_id, job in enumerate(jobs):
        prev_end = None
        for op_idx, (machine, duration) in enumerate(job):
            start = model.NewIntVar(0, horizon, f'start_{job_id}_{op_idx}')
            end = model.NewIntVar(0, horizon, f'end_{job_id}_{op_idx}')
            interval = model.NewIntervalVar(start, duration, end, f'interval_{job_id}_{op_idx}')
            all_ops.append((job_id, op_idx, machine, start, end))
            intervals_per_machine[machine].append(interval)
            starts.append(start)
            ends.append(end)
            
            if prev_end is not None:
                model.Add(start >= prev_end)
            prev_end = end

    for machine_id, machine_intervals in enumerate(intervals_per_machine):
        if machine_intervals:
            model.AddNoOverlap(machine_intervals)
    
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    
    # ✅ Configuració crítica de temps límit i workers
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8  # Parallel search
    solver.parameters.log_search_progress = True  # Important per veure progrés
    
    print(f"Iniciant resolució amb límit de {time_limit_seconds} segons...")
    
    try:
        status = solver.Solve(model)
        print(f"Status: {solver.StatusName(status)}")
        
        if status == cp_model.OPTIMAL:
            print("S'ha trobat la solució òptima!")
        elif status == cp_model.FEASIBLE:
            print("S'ha trobat una solució factible (però pot no ser òptima)")
        elif status == cp_model.INFEASIBLE:
            print("El problema no té solució")
        elif status == cp_model.MODEL_INVALID:
            print("El model és invàlid")
        else:
            print(f"Altres status: {status}")
            
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = []
            for job_id, op_idx, machine, start_var, end_var in all_ops:
                start = solver.Value(start_var)
                end = solver.Value(end_var)
                schedule.append((job_id, op_idx, machine, start, end))
            return schedule, solver.Value(makespan)
        else:
            return None, None
    except Exception as e:
        print(f"Error durant la resolució: {e}")
        return None, None


# Codi principal
if __name__ == "__main__":
    
    num_jobs = 7
    num_machines = 2
    num_operations = 100
    min_duration = 1
    max_duration = 10

    # Genera un problema de mida moderada
    print("Generant problema aleatori...")
    jobs = generate_random_jobs(num_jobs, num_machines, num_operations, min_duration, max_duration)
    print(f"Problema generat: {len(jobs)} jobs, {len(jobs[0])} operacions per job")
    
    start_time = time.time()
    schedule, makespan = cp_solver(jobs, time_limit_seconds=60)
    total_time = time.time() - start_time
    
    fitxer = r"C:\Users\EduardGonzalvoGelabe\OneDrive - Capitole Consulting\Documentos\codis\genetic-algorithms\job_scheduling\results.txt"
    with open(fitxer, "a", encoding="utf-8") as f:
        f.write(f"CP-SAT\t\t{num_jobs}\t{num_machines}\t{num_operations}\t{makespan}\t{total_time:.2f}\n")

    if schedule:
        print(f"Operacions: {len(schedule)}")
        print(f"Makespan: {makespan}")
        print(f"Temps total d'execució: {total_time:.2f} segons")
        print("Mostrant diagrama de Gantt...")
        plot_gantt(schedule)
    else:
        print("No s'ha trobat cap solució en el temps límit.")
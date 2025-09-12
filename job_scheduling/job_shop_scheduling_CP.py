from matplotlib import pyplot, patches
from ortools.sat.python import cp_model


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


def cp_solver(jobs):
    model = cp_model.CpModel()
    num_machines = max(max(op[0] for op in job) for job in jobs) + 1
    all_ops = []
    intervals_per_machine = [[] for _ in range(num_machines)]
    starts = []
    ends = []

    for job_id, job in enumerate(jobs):
        prev_end = None
        for op_idx, (machine, duration) in enumerate(job):
            start = model.NewIntVar(0, 10000, f'start_{job_id}_{op_idx}')
            end = model.NewIntVar(0, 10000, f'end_{job_id}_{op_idx}')
            interval = model.NewIntervalVar(start, duration, end, f'interval_{job_id}_{op_idx}')
            all_ops.append((job_id, op_idx, machine, start, end))
            intervals_per_machine[machine].append(interval)
            starts.append(start)
            ends.append(end)
            model.Add(end == start + duration)
            if prev_end is not None:
                model.Add(start >= prev_end)
            prev_end = end

    for machine_intervals in intervals_per_machine:
        if machine_intervals:
            model.AddNoOverlap(machine_intervals)
    
    makespan = model.NewIntVar(0, 10000, 'makespan')
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = []
        for job_id, op_idx, machine, start_var, end_var in all_ops:
            start = solver.Value(start_var)
            end = solver.Value(end_var)
            schedule.append((job_id, op_idx, machine, start, end))
        return schedule, solver.Value(makespan)
    else:
        return None, None


jobs = [[(0, 3), (1, 2), (2, 2)], [(0, 2), (2, 1), (1, 4)], [(1, 4), (2, 3), (0, 2)]]
jobs = [[(9, 2), (7, 2), (2, 6), (8, 2), (1, 9)], [(4, 5), (8, 6), (5, 5), (5, 3), (2, 10)], [(4, 10), (5, 6), (1, 3), (3, 1), (9, 8)], [(9, 5), (6, 3), (3, 8), (2, 9), (0, 2)], [(1, 4), (4, 5), (7, 4), (3, 8), (2, 1)], [(6, 8), (9, 4), (1, 5), (4, 10), (5, 9)], [(8, 10), (1, 10), (7, 10), (9, 4), (0, 8)], [(6, 6), (6, 8), (2, 9), (4, 1), (4, 4)], [(8, 9), (2, 10), (4, 7), (6, 3), (3, 1)], [(6, 1), (5, 5), (3, 2), (1, 8), (0, 7)], [(7, 8), (3, 7), (3, 1), (0, 10), (7, 1)], [(8, 4), (8, 7), (5, 9), (7, 5), (2, 4)], [(5, 8), (1, 2), (8, 1), (8, 4), (4, 6)], [(3, 8), (8, 9), (5, 6), (4, 10), (4, 1)], [(6, 7), (0, 8), (9, 6), (1, 3), (1, 6)], [(9, 9), (3, 2), (5, 10), (2, 3), (4, 7)], [(6, 4), (7, 9), (0, 10), (5, 4), (9, 2)], [(3, 7), (3, 9), (9, 5), (2, 3), (2, 10)], [(7, 3), (9, 5), (0, 6), (5, 2), (2, 8)], [(2, 7), (4, 2), (9, 1), (9, 5), (7, 2)]]
jobs = [[(5, 2), (3, 2), (8, 5), (2, 1), (3, 5)], [(2, 4), (8, 2), (8, 1), (2, 1), (9, 1)], [(6, 3), (1, 2), (5, 2), (5, 5), (9, 4)], [(6, 3), (5, 2), (4, 1), (8, 4), (1, 1)], [(9, 5), (2, 1), (6, 1), (1, 3), (2, 3)], [(0, 3), (9, 2), (2, 3), (0, 1), (2, 5)], [(3, 4), (1, 1), (2, 5), (3, 1), (3, 5)], [(3, 2), (5, 4), (6, 2), (5, 1), (2, 2)], [(4, 2), (2, 3), (1, 1), (4, 3), (2, 2)], [(4, 3), (9, 1), (3, 2), (4, 5), (7, 1)], [(3, 3), (9, 3), (0, 5), (5, 1), (0, 5)], [(9, 1), (6, 1), (7, 4), (2, 3), (1, 1)], [(7, 1), (2, 2), (6, 2), (9, 5), (9, 5)], [(1, 2), (4, 1), (4, 3), (4, 3), (4, 4)], [(0, 1), (0, 2), (9, 2), (3, 2), (5, 3)], [(1, 2), (0, 2), (7, 3), (3, 3), (2, 3)], [(2, 2), (0, 1), (8, 3), (4, 1), (7, 5)], [(7, 3), (7, 1), (2, 4), (7, 3), (0, 5)], [(6, 4), (8, 4), (4, 4), (0, 2), (9, 3)], [(3, 4), (1, 4), (8, 3), (6, 3), (6, 2)], [(3, 4), (5, 1), (5, 5), (5, 3), (4, 5)], [(8, 4), (5, 3), (6, 1), (1, 3), (6, 3)], [(3, 4), (2, 2), (2, 5), (2, 1), (6, 5)], [(0, 4), (4, 1), (1, 5), (8, 1), (3, 4)], [(4, 3), (4, 1), (2, 1), (6, 3), (9, 4)], [(1, 1), (9, 1), (9, 4), (5, 4), (7, 3)], [(6, 2), (3, 3), (7, 2), (4, 5), (1, 4)], [(9, 4), (9, 5), (3, 1), (4, 4), (4, 4)], [(7, 1), (1, 2), (1, 4), (8, 1), (2, 1)], [(6, 1), (5, 2), (6, 5), (0, 2), (6, 5)], [(9, 4), (7, 3), (0, 1), (9, 3), (1, 3)], [(0, 3), (8, 3), (7, 5), (0, 4), (8, 1)], [(7, 5), (9, 1), (2, 3), (1, 1), (6, 5)], [(4, 4), (6, 3), (7, 1), (4, 5), (2, 4)], [(6, 5), (7, 4), (0, 4), (5, 5), (3, 3)], [(8, 2), (5, 3), (9, 2), (9, 4), (2, 4)], [(0, 3), (9, 4), (9, 2), (8, 2), (4, 4)], [(0, 3), (0, 2), (4, 4), (1, 5), (7, 5)], [(2, 2), (4, 1), (2, 1), (8, 5), (1, 3)], [(6, 1), (8, 2), (1, 2), (3, 4), (5, 3)], [(0, 5), (9, 5), (7, 3), (1, 3), (8, 5)], [(1, 2), (2, 5), (5, 3), (9, 4), (7, 2)], [(8, 1), (8, 1), (5, 1), (1, 4), (9, 5)], [(4, 1), (8, 5), (5, 1), (7, 4), (8, 5)], [(7, 2), (5, 1), (0, 1), (6, 3), (4, 3)], [(5, 4), (8, 2), (5, 2), (4, 3), (4, 4)], [(2, 4), (5, 5), (7, 3), (8, 2), (8, 5)], [(0, 1), (4, 2), (3, 4), (3, 5), (2, 3)], [(4, 4), (3, 1), (5, 1), (2, 1), (3, 4)], [(3, 1), (2, 5), (4, 5), (0, 2), (1, 1)], [(1, 5), (4, 2), (1, 2), (2, 4), (6, 4)], [(2, 1), (8, 1), (0, 4), (3, 4), (0, 3)], [(7, 3), (1, 4), (1, 2), (8, 4), (9, 4)], [(0, 1), (2, 5), (4, 5), (2, 4), (1, 1)], [(7, 4), (9, 3), (1, 1), (1, 1), (3, 3)], [(5, 3), (8, 3), (6, 3), (4, 1), (1, 5)], [(9, 2), (5, 5), (1, 5), (6, 2), (3, 2)], [(5, 3), (4, 3), (5, 4), (1, 3), (2, 5)], [(8, 3), (5, 1), (2, 4), (9, 2), (6, 3)], [(4, 1), (5, 4), (6, 2), (0, 4), (0, 4)]]

schedule, makespan = cp_solver(jobs)
if schedule:
    print("Best Schedule:", schedule)
    print("Makespan:", makespan)
    plot_gantt(schedule)
else:
    print("No s'ha trobat cap solució.")

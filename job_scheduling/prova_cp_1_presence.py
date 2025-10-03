from ortools.sat.python import cp_model

model = cp_model.CpModel()

horizon = 20

# Exemple: dues activitats amb mínim/màxim/duaració i nombre màxim d'instàncies
activities = {
    "A": {"min_slots": 2, "max_slots": 4, "duration": 3, "earliest": 0, "latest": 20},
    "B": {"min_slots": 1, "max_slots": 3, "duration": 2, "earliest": 0, "latest": 20}
}

# Diccionaris per guardar referències a les variables
start_vars = {}
end_vars = {}
presence_vars = {}
intervals = []
count_vars = {}

for a, data in activities.items():
    dur = data["duration"]
    max_slots = data["max_slots"]
    presences = []
    for i in range(max_slots):
        # Crea start i end amb domini global; vinculem la relació només si està present
        s = model.NewIntVar(0, horizon, f"start_{a}_{i}")
        e = model.NewIntVar(0, horizon, f"end_{a}_{i}")
        p = model.NewBoolVar(f"pres_{a}_{i}")
        interval = model.NewOptionalIntervalVar(s, dur, e, p, f"interval_{a}_{i}")

        # Enllaça end = start + dur només si p == 1
        model.Add(e == s + dur).OnlyEnforceIf(p)

        # Opcional: imposar finestres de temps només quan està present
        model.Add(s >= data["earliest"]).OnlyEnforceIf(p)
        model.Add(e <= data["latest"]).OnlyEnforceIf(p)

        start_vars[(a, i)] = s
        end_vars[(a, i)] = e
        presence_vars[(a, i)] = p
        intervals.append(interval)
        presences.append(p)

    # count variable que compta quantes instàncies actives hi ha d'aquesta activitat
    c = model.NewIntVar(data["min_slots"], data["max_slots"], f"count_{a}")
    model.Add(c == sum(presences))
    count_vars[a] = c

# Restricció: cap solapament entre tots els intervals (les opcions opcionals són acceptades)
model.AddNoOverlap(intervals)

# Objectiu d'exemple: maximitzar el nombre total d'instàncies programades
model.Minimize(sum(count_vars.values()))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 10
status = solver.Solve(model)

print("Status:", solver.StatusName(status))
for a in activities:
    print(f"{a}: count = {solver.Value(count_vars[a])}")
    for i in range(activities[a]["max_slots"]):
        p = presence_vars[(a, i)]
        if solver.Value(p):
            s_val = solver.Value(start_vars[(a, i)])
            e_val = solver.Value(end_vars[(a, i)])
            print(f"  {a}_{i}: {s_val} -> {e_val}")

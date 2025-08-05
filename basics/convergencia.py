max_generations = 1000
patience = 50
diversity_threshold = 0.01

best_fitness_history = []
stagnant_generations = 0

for generation in range(max_generations):
    # evolució normal: selecció, crossover, mutació...
    
    best_fitness = max(fitnesses)
    best_fitness_history.append(best_fitness)

    # comprovem estancament
    if generation > 0 and best_fitness_history[-1] == best_fitness_history[-2]:
        stagnant_generations += 1
    else:
        stagnant_generations = 0

    # diversitat
    avg = sum(fitnesses) / len(fitnesses)
    diversity = math.sqrt(sum((f - avg)**2 for f in fitnesses) / len(fitnesses))

    if stagnant_generations >= patience or diversity < diversity_threshold:
        print(f"S'atura a la generació {generation}")
        break

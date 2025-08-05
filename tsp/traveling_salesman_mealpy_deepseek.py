import numpy as np
from mealpy import Problem, FloatVar, PSO
import matplotlib.pyplot as plt

# Punts de mostra
sample_points = [(38, 83), (54, 13), (19, 2), (96, 30), (52, 7), (48, 32), (53, 92),
                 (70, 6), (92, 27), (96, 63), (8, 56), (67, 100), (59, 79), (48, 69),
                 (35, 79), (64, 44), (14, 79), (2, 8), (1, 6), (47, 95)]

# Funció per calcular la distància euclidiana
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Funció per calcular la distància total d'una ruta
def total_distance(route, points):
    distance = 0
    for i in range(len(route)):
        from_point = points[route[i]]
        to_point = points[route[(i+1)%len(route)]]
        distance += euclidean_distance(from_point, to_point)
    return distance

# Funció objectiu
def objective_function(solution, points):
    # Convertim la solució contínua a una permutació
    ranked = sorted(range(len(solution)), key=lambda x: solution[x])
    return total_distance(ranked, points)

# Definim el problema correctament
bounds = [FloatVar(lb=0.0, ub=1.0, name=f"x{i}") for i in range(len(sample_points))]
problem = Problem(
    bounds=bounds,
    minmax="min",
    obj_func=lambda sol: objective_function(sol, sample_points)
)

# Configuració de PSO
epoch = 1000
pop_size = 200
c1 = 2.0
c2 = 2.0
w = 0.7

model = PSO.OriginalPSO(epoch, pop_size, c1, c2, w)
best_position, best_fitness = model.solve(problem)

# Obtenim la millor ruta
ranked = sorted(range(len(best_position)), key=lambda x: best_position[x])
optimal_route = [sample_points[i] for i in ranked]
optimal_distance = best_fitness

# Afegim el primer punt al final per tancar el cicle
optimal_route.append(optimal_route[0])

# Resultats
print("Millor ruta trobada:")
for i, point in enumerate(optimal_route[:-1]):
    print(f"{i+1}: {point}")
print(f"\nDistància total: {optimal_distance:.2f}")

# Visualització
plt.figure(figsize=(12, 8))
x = [point[0] for point in optimal_route]
y = [point[1] for point in optimal_route]

plt.plot(x, y, 'o-', markersize=8, linewidth=2)
plt.title('Millor ruta TSP', fontsize=14)
plt.xlabel('Coordenada X', fontsize=12)
plt.ylabel('Coordenada Y', fontsize=12)

# Numerem els punts per veure l'ordre
for i, (xi, yi) in enumerate(optimal_route[:-1]):
    plt.text(xi, yi, str(i+1), color='red', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
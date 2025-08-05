import numpy as np
from mealpy import FloatVar, PSO
import matplotlib.pyplot as plt

# Punts del problema
sample_points = [(38, 83), (54, 13), (19, 2), (96, 30), (52, 7), (48, 32), (53, 92),
                 (70, 6), (92, 27), (96, 63), (8, 56), (67, 100), (59, 79), (48, 69),
                 (35, 79), (64, 44), (14, 79), (2, 8), (1, 6), (47, 95)]
n_ciutats = len(sample_points)

# Matriu de distàncies
distancias = np.zeros((n_ciutats, n_ciutats))
for i in range(n_ciutats):
    for j in range(n_ciutats):
        if i != j:
            x1, y1 = sample_points[i]
            x2, y2 = sample_points[j]
            distancias[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Funció objectiu
def tsp_objectiu(solucio_cont):
    # Convertim solució contínua a permutació (rank-order encoding)
    permutacio = np.argsort(solucio_cont)
    
    # Calculem distància total de la ruta
    distancia_total = 0
    for k in range(n_ciutats - 1):
        i = permutacio[k]
        j = permutacio[k + 1]
        distancia_total += distancias[i, j]
    
    # Tornem al punt inicial
    distancia_total += distancias[permutacio[-1], permutacio[0]]
    return distancia_total

# Definim el problema d'optimització (CORRECCIÓ: 'bounds' en lloc de 'limits')
problema = {
    "obj_func": tsp_objectiu,  # Nom canviar per versions recents
    "bounds": FloatVar(lb=[0.0] * n_ciutats, ub=[1.0] * n_ciutats),
    "minmax": "min",
}

# Configurem i executem l'algorisme PSO
model = PSO.OriginalPSO(epoch=1000, pop_size=200)
millor_solucio = model.solve(problema)

# Processem la millor solució trobada
permutacio_final = np.argsort(millor_solucio.solution)
distancia_final = millor_solucio.target.fitness

# Reconstruïm la ruta (tancant el cicle)
ruta_final = permutacio_final.tolist()
ruta_final.append(ruta_final[0])

# Resultats
print(f"\nDistància mínima trobada: {distancia_final:.2f}")
print("Ruta òptima (índexs de ciutats):")
print(' -> '.join(map(str, ruta_final)))

# Visualització
plt.figure(figsize=(10, 6))
x_vals = [p[0] for p in sample_points]
y_vals = [p[1] for p in sample_points]

# Dibuixem punts
plt.scatter(x_vals, y_vals, c='red', s=100, zorder=5)

# Dibuixem ruta
ruta_coords = [sample_points[i] for i in ruta_final]
ruta_x = [p[0] for p in ruta_coords]
ruta_y = [p[1] for p in ruta_coords]
plt.plot(ruta_x, ruta_y, 'b-', linewidth=1.5, zorder=3)

# Anotem índexs
for i, (x, y) in enumerate(sample_points):
    plt.annotate(str(i), (x+1, y+1), fontsize=9)

plt.title("Ruta Òptima TSP")
plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
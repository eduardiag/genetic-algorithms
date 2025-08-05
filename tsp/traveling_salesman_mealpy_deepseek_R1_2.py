import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from mealpy import PermutationVar, SA
import time

# Punts del problema
sample_points = [(38, 83), (54, 13), (19, 2), (96, 30), (52, 7), (48, 32), (53, 92),
                 (70, 6), (92, 27), (96, 63), (8, 56), (67, 100), (59, 79), (48, 69),
                 (35, 79), (64, 44), (14, 79), (2, 8), (1, 6), (47, 95)]
n_ciutats = len(sample_points)

# 1. Càlcul de la matriu de distàncies
distancias = np.zeros((n_ciutats, n_ciutats))
for i in range(n_ciutats):
    for j in range(n_ciutats):
        if i != j:
            x1, y1 = sample_points[i]
            x2, y2 = sample_points[j]
            distancias[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 2. Funció objectiu per TSP
def tsp_objectiu(permutacio):
    dist_total = 0
    for k in range(len(permutacio) - 1):
        dist_total += distancias[permutacio[k], permutacio[k+1]]
    # Tornar al punt inicial
    dist_total += distancias[permutacio[-1], permutacio[0]]
    return dist_total

# 3. Càlcul de la solució òptima real (per força bruta, només per n=20)
def trobar_optima():
    millor_dist = float('inf')
    millor_ruta = None
    ciutats = list(range(n_ciutats))
    
    # Provem totes les permutacions
    for i, perm in enumerate(permutations(ciutats)):
        if i % 1000000 == 0:  # Mostrar progrés cada 1M de permutacions
            print(f"Processades {i/1e6:.1f}M permutacions...")
        dist = tsp_objectiu(perm)
        if dist < millor_dist:
            millor_dist = dist
            millor_ruta = perm
    return list(millor_ruta), millor_dist

# 4. Optimització amb Simulated Annealing
def optimitzar_amb_sa():
    problema = {
        "obj_func": tsp_objectiu,
        "bounds": PermutationVar(valid_set=list(range(n_ciutats))),
        "minmax": "min"
    }

    model = SA.OriginalSA(epoch=5000, pop_size=50, max_sub_iter=15, t0=1000, cooling_rate=0.95)
    return model.solve(problema)

# 5. Visualització de resultats
def visualitzar_ruta(ruta, distancia, titol):
    plt.figure(figsize=(12, 8))
    x = [p[0] for p in sample_points]
    y = [p[1] for p in sample_points]
    
    # Dibuixar punts
    plt.scatter(x, y, c='red', s=150, zorder=5)
    
    # Dibuixar ruta
    ruta_coords = [sample_points[i] for i in ruta]
    ruta_x = [p[0] for p in ruta_coords]
    ruta_y = [p[1] for p in ruta_coords]
    plt.plot(ruta_x, ruta_y, 'b-', linewidth=2, zorder=3)
    
    # Anotar índexs i distàncies
    for i, (x, y) in enumerate(sample_points):
        plt.annotate(f"{i}", (x+1, y+1), fontsize=11, fontweight='bold')
    
    plt.title(f"{titol} (Distància: {distancia:.2f})", fontsize=14)
    plt.xlabel("Coordenada X", fontsize=12)
    plt.ylabel("Coordenada Y", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

# Execució principal
if __name__ == "__main__":
    print("Calculant solució òptima real (força bruta)...")
    start_time = time.time()
    ruta_optima, distancia_optima = trobar_optima()
    temps_optima = time.time() - start_time
    print(f"\nSolució òptima trobada en {temps_optima:.2f} segons")
    print(f"Distància òptima: {distancia_optima:.2f}")
    print(f"Ruta òptima: {ruta_optima}")
    
    print("\nOptimitzant amb Simulated Annealing...")
    start_time = time.time()
    solucio_sa = optimitzar_amb_sa()
    temps_sa = time.time() - start_time
    ruta_sa = solucio_sa.solution
    distancia_sa = solucio_sa.target.fitness
    
    print(f"\nSA completat en {temps_sa:.2f} segons")
    print(f"Millor distància SA: {distancia_sa:.2f}")
    print(f"Ruta SA: {ruta_sa}")
    
    # Visualitzar resultats
    visualitzar_ruta(ruta_optima + [ruta_optima[0]], distancia_optima, "Solució Òptima (Força Bruta)")
    visualitzar_ruta(ruta_sa + [ruta_sa[0]], distancia_sa, "Solució amb Simulated Annealing")
    plt.show()

# Resultats de la solució òptima
print("\nRESULTAT FINAL")
print(f"Distància mínima real: {distancia_optima:.2f}")
print("Ruta òptima completa:")
ruta_optima_str = ' -> '.join(map(str, ruta_optima)) + f" -> {ruta_optima[0]}"
print(ruta_optima_str)
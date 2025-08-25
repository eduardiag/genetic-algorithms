from matplotlib import pyplot
import random
import numpy
import math

# Parameters
n_cities = 300
dim_grid = 100

# Generació de punts aleatoris
sample_points = [(random.randint(0, dim_grid), random.randint(0, dim_grid)) for _ in range(n_cities)]

print("Running TSP with parameters:")
print(f" - Number of cities: {n_cities}")
print(f" - Grid size: {dim_grid}")

dist = numpy.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        dx = sample_points[i][0] - sample_points[j][0]
        dy = sample_points[i][1] - sample_points[j][1]
        dist[i][j] = math.sqrt(dx * dx + dy * dy)

def two_opt(tour, max_passes=10):
    best = list(map(int, tour))
    best_len = 0
    for i in range(len(tour)):
        a = int(tour[i])
        b = int(tour[(i + 1) % len(tour)])
        best_len += dist[a, b]
    n = len(best)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = best[i - 1], best[i]
                c, d = best[k], best[(k + 1) % n]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < -1e-9:
                    best[i:k + 1] = reversed(best[i:k + 1])
                    best_len += delta
                    improved = True
        if not improved:
            break
    return best, best_len

route = list(range(n_cities))
random.shuffle(route)
best_route, best_distance = two_opt(route, max_passes=10)

print(f"Best distance: {best_distance:.2f}")
print(f"Route (primeres 20): {best_route}")

x = [sample_points[i][0] for i in best_route] + [sample_points[best_route[0]][0]]
y = [sample_points[i][1] for i in best_route] + [sample_points[best_route[0]][1]]

pyplot.figure(figsize=(8, 8))
pyplot.plot(x, y, 'b-', linewidth=2, alpha=0.7, label='Tour')
pyplot.scatter([p[0] for p in sample_points], [p[1] for p in sample_points], c='red', s=50, zorder=5)
pyplot.scatter(x[0], y[0], c='green', s=100, marker='s', zorder=6, label='Start')
pyplot.title(f'TSP Solution - Distance: {best_distance:.2f}')
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.grid(True, alpha=0.3)
pyplot.legend()
pyplot.tight_layout()
pyplot.show()
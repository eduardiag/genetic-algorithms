
from matplotlib import pyplot
import numpy as np




sample_points = [(14, 17), (69, 76), (76, 17), (81, 52), (63, 53), (14, 77), (33, 27), (93, 80), (16, 50), (52, 99), (39, 36), (54, 36), (28, 74), (39, 47), (37, 29), (14, 65), (30, 2), (5, 6), (25, 16), (98, 32), (55, 42), (12, 86), (13, 94), (4, 26), (87, 44), (48, 17), (38, 93), (39, 63), (9, 72), (0, 94), (78, 83), (79, 66), (67, 21), (73, 5), (3, 32), (32, 7), (93, 93), (96, 77), (43, 71), (15, 46), (97, 51), (51, 24), (1, 26), (86, 86), (3, 85), (54, 25), (15, 90), (49, 12), (82, 96), (82, 12), (34, 9), (70, 6), (93, 78), (98, 88), (41, 32), (88, 18), (57, 21), (91, 69), (58, 17), (7, 33), (25, 61), (57, 25), (17, 55), (92, 81), (36, 14), (17, 45), (17, 25), (3, 83), (82, 29), (82, 82), (84, 28), (99, 6), (6, 33), (7, 32), (46, 67), (24, 71), (24, 14), (42, 93), (3, 98), (98, 34), (33, 95), (82, 45), (13, 52), (66, 69), (54, 41), (26, 80), (31, 58), (87, 51), (36, 60), (95, 46), (11, 0), (98, 50), (37, 76), (24, 92), (53, 96), (14, 35), (35, 25), (3, 74), (49, 53), (0, 77)]
best = [92, 94, 9, 77, 26, 80, 93, 85, 21, 46, 22, 78, 29, 44, 67, 99, 97, 28, 5, 15, 75, 12, 86, 60, 62, 82, 8, 39, 65, 95, 23, 42, 34, 72, 59, 73, 66, 0, 17, 90, 16, 35, 50, 64, 18, 76, 96, 6, 14, 10, 54, 25, 47, 41, 45, 61, 56, 58, 32, 51, 33, 49, 71, 55, 2, 70, 68, 81, 24, 19, 79, 89, 91, 40, 87, 3, 57, 37, 52, 7, 63, 43, 53, 36, 48, 69, 30, 1, 83, 31, 4, 20, 11, 84, 98, 13, 88, 27, 74, 38]
print(len(sample_points), len(best))    

sample_points = [(38, 83), (54, 13), (19, 2), (96, 30), (52, 7), (48, 32), (53, 92), (70, 6), (92, 27), (96, 63), (8, 56), (67, 100), (59, 79), (48, 69), (35, 79), (64, 44), (14, 79), (2, 8), (1, 6), (47, 95)]
best = [14, 16, 10, 17, 18, 2, 4, 1, 7, 8, 3, 9, 15, 5, 13, 12, 11, 6, 19, 0]
print(len(sample_points), len(best))   


def euclidean_distance(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return np.sqrt(dx * dx + dy * dy)

n_cities = len(sample_points)

# Imprimir resultats:
dist = sum(euclidean_distance(sample_points[best[i]], sample_points[best[(i + 1) % n_cities]]) for i in range(n_cities))
print(best)
print(dist)

# Visualització del millor recorregut
x_coords = [sample_points[point][0] for point in best] + [sample_points[best[0]][0]]
y_coords = [sample_points[point][1] for point in best] + [sample_points[best[0]][1]]

pyplot.figure(figsize=(8, 6))
pyplot.plot(x_coords, y_coords, marker='o', linestyle='-')
for p in best:
    pyplot.text(sample_points[p][0], sample_points[p][1], p, fontsize=9)

pyplot.title(f'Tour aleatori de {n_cities} punts enters')
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.grid(True)
pyplot.axis('equal')
pyplot.show()

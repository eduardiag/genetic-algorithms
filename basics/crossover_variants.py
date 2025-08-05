
# ------------------------------------------

parents = ["001100", "010000", "000111", "110001"]
crossover_rate = 0.6

# ------------------------------------------

import random

def one_point_crossover(p1, p2):
    point = random.randint(1, len(p1)-1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def two_point_crossover(p1, p2):
    pt1, pt2 = sorted(random.sample(range(1, len(p1)), 2))
    return (p1[:pt1] + p2[pt1:pt2] + p1[pt2:], 
            p2[:pt1] + p1[pt1:pt2] + p2[pt2:])

def uniform_crossover(p1, p2, prob=0.5):
    c1, c2 = [], []
    for g1, g2 in zip(p1, p2):
        if random.random() < prob:
            c1.append(g1)
            c2.append(g2)
        else:
            c1.append(g2)
            c2.append(g1)
    return c1, c2


offspring = []
for i in range(0, len(parents)-1, 2):
    p1, p2 = parents[i], parents[i+1]
    if random.random() < crossover_rate:
        child1, child2 = one_point_crossover(p1, p2)
    else:
        child1, child2 = p1[:], p2[:]
    offspring.extend([child1, child2])

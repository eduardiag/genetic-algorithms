
import random


# Bit flip mutation: Inverteix un bit amb certa probabilitat p_mut.
def mutate(individual, p_mut=0.01):
    return [bit if random.random() > p_mut else 1 - bit for bit in individual]

# Uniform integer mutation: Canvia un gen a un altre valor dins del domini permès.
def mutate(individual, p_mut=0.1, low=0, high=10):
    return [gene if random.random() > p_mut else random.randint(low, high)
            for gene in individual]

# Gaussian mutation: Afegeix soroll normal N(0, σ) al gen.
def mutate(individual, p_mut=0.1, sigma=0.1):
    return [gene + random.gauss(0, sigma) if random.random() < p_mut else gene
            for gene in individual]



"""
❓ Extensions i variacions
- Mutació adaptativa: la taxa canvia amb el temps (e.g., decreix si no hi ha millora).
- Self-adaptive mutation: el genoma inclou la seva pròpia taxa de mutació.
- Crowding / niching: mutació combinada amb tècniques per preservar solucions diverses.
- Non-uniform mutation: la magnitud de la mutació decreix amb el número de generacions.
"""
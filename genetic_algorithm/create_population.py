from typing import List, Dict, Callable, Optional
import random


def create_population(pop_size, num_points, seeding, seeding_size):
    population = []
    if seeding:
        population.append(seeding[:])
        for _ in range(seeding_size-1):
            individual = seeding[:]
            random.shuffle(individual)
            population.append(individual)

    while len(population) < pop_size:
        population.append(random.sample(range(num_points), num_points))
    return population
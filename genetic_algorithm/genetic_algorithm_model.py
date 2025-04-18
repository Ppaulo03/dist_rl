from .routing_utils import route_distance_matrix, precompute_distance_matrix
from .select_parent import PARENT_SELECTION, select_parents
from .mutation import MUTATION_SELECTION, mutate
from .crossover import CROSSOVER_STRATEGY, crossover
from .create_population import create_population

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Callable, Tuple, Sequence
from loguru import logger
import numpy as np
import random


def genetic_algorithm(  points:List[Tuple[float, float]], origin:Tuple[float, float], 
                        generations:int=300, pop_size:int=150, mutation_rate:float=0.02,
                        early_stop:int = 50, elitism:bool=True,
                        route_distance_matrix:Callable[[Sequence[int], np.ndarray], float]=route_distance_matrix,
                        seeding:Optional[List[int]]=None,
                        selection_method:PARENT_SELECTION=PARENT_SELECTION.TOURNAMENT, k:int=8,
                        crossover_strategy:CROSSOVER_STRATEGY=CROSSOVER_STRATEGY.PMX,
                        mutation_strategy:MUTATION_SELECTION=MUTATION_SELECTION.SWAP,
                        use_threads:bool=False, max_threads:int=4,
                        seed:Optional[int]=None, verbose=False
                        ) -> List[tuple[float, float]]:
    
    '''
    Genetic Algorithm for solving the Traveling Salesman Problem (TSP).

    Args:
        points (List[Tuple[float, float]]): 
            List of coordinates for all points (cities) to visit.
        
        origin (Tuple[float, float]): 
            Coordinates of the starting point.
        
        generations (int): 
            Number of generations to run the algorithm. Default is 300.
        
        pop_size (int): 
            Size of the population. Default is 150.
        
        mutation_rate (float): 
            Probability of mutation for each individual in the population. Default is 0.02.

        early_stop (int): 
            Number of generations without improvement before stopping. Default is 50.

        elitism (bool): 
            Whether to keep the best individual from the previous generation. Default is True.

        route_distance_matrix (Callable[[Sequence[int], ndarray], float]): 
            Function to calculate the distance of a route.

        seeding (Optional[List[int]]): 
            Optional route to seed the population.

        selection_method (PARENT_SELECTION): 
            The selection strategy to apply. Options are:
                - TOURNAMENT: Selects the best individual from a random subset of the population.
                - RANK: Selects individuals based on their fitness ranking.
                - ROULETTE: Selects individuals proportionally to their fitness.
                - TOP_HALF: Selects individuals from the top half of the population.
                - RANDOM: Selects individuals randomly.

        k (int): 
            Parameter used by some selection strategies on selection method. Default is 8.

        crossover_strategy (CROSSOVER_STRATEGY): 
            The crossover strategy to use. Options:
                - PMX: Partially Mapped Crossover, Creates offspring by partially mapping genes from both parents.
                - OX: Order Crossover, Creates offspring by preserving the order of genes from both parents.
                - CX: Cycle Crossover, Creates offspring by forming cycles between the two parents.
                - ERX: Edge Recombination Crossover, Creates offspring by preserving edges between genes.
                - PMXM: Modified PMX Crossover, A variation of PMX that uses a different mapping strategy.
                - POS: Position-Based Crossover, Creates offspring by selecting genes based on their positions in the parents.

        mutation_strategy (MUTATION_SELECTION): 
            The mutation strategy to use. If None, a random strategy will be selected. Options:
                - SWAP: Swaps the position of two cities in the route.
                - INVERSION: Reverses the order of a subsequence within the route.
                - INSERTION: Removes one city from the route and reinserts it at another position.
                - ROTATION: Rotates a subsequence of the route (similar to inversion but always applied).
                - NON_ADJACENT_SWAP: Swaps two cities only if they are not adjacent.
                - SCRAMBLE: Randomly shuffles the order of cities in a subsequence.
                - CIRCULAR_SHIFT: Performs a circular shift on a subsequence, moving elements by one position.

        use_threads (bool): 
            Whether to use threading for parallel processing. Default is False.

        max_threads (int):
            Maximum number of threads to use if threading is enabled. Default is 4.

        seed (Optional[int]): 
            Random seed for reproducibility. Default is None.

    Returns:
        List[Tuple[float, float]]: 
            The best route found by the algorithm.
    '''
    
    rng = random.Random(seed) if seed is not None else random
    num_points = len(points)
    population = create_population(pop_size, num_points, seeding)
    best_route = None
    best_distance = float('inf')
    dist_matrix = precompute_distance_matrix(points, origin)

    no_improve_counter = 0
    
    for generation in range(generations):   

        if use_threads:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(select_parents, population, dist_matrix, route_distance_matrix, strategy=selection_method, k=k)]
                parents = futures[0].result()
        else:
            parents = select_parents(population, dist_matrix, route_distance_matrix, strategy=selection_method, k=k)

        # Elitismo: guardar o melhor
        elite = min(parents, key=lambda r: route_distance_matrix(r, dist_matrix))
        elite_distance = route_distance_matrix(elite, dist_matrix)

        if elite_distance < best_distance:
            best_distance = elite_distance
            best_route = elite
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if early_stop > 0 and no_improve_counter >= early_stop:
            if verbose:
                logger.info(f"Parou na geração {generation} por early stopping. Melhor distância: {best_distance:.2f} km")
            return best_route
    
        # Preenche população com filhos
        next_population = []

        def cross_and_mutate(p1, p2):
            child1, child2 = crossover(p1, p2, crossover_strategy)
            child1 = mutate(child1, mutation_rate, mutation_strategy)
            child2 = mutate(child2, mutation_rate, mutation_strategy)
            return child1, child2
        
        if use_threads:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
        
                futures = []
                for _ in range(pop_size // 2):
                    p1, p2 = rng.sample(parents, 2)
                    futures.append(executor.submit(cross_and_mutate, p1, p2))

                for future in as_completed(futures):
                    child1, child2 = future.result()
                    next_population.append(child1)
                    next_population.append(child2)

        else:
            while len(next_population) < pop_size:
                p1, p2 = rng.sample(parents, 2)
                child1, child2 = cross_and_mutate(p1, p2)
                next_population.append(child1)
                next_population.append(child2)
        
        # Se elitismo estiver ativado, passa o elite pra próxima geração
        if elitism and elite not in next_population:
            if len(next_population) >= pop_size:
                next_population[rng.randint(0, pop_size-1)] = elite  # substitui aleatoriamente
            else:
                next_population.append(elite)

        population = next_population

        # Print a cada 50 gerações
        if verbose and generation % 50 == 0:
            logger.info(f"Geração {generation}: Melhor até agora = {best_distance:.2f} km")
        
    if verbose:
        logger.info(f"Algoritmo finalizou todas as gerações. Melhor distância encontrada: {best_distance:.2f} km")
    return best_route
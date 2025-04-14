from typing import List, Tuple, Dict, Callable, Optional, Sequence
from numpy import ndarray
import numpy as np

import random
import enum


def _tournament_selection(scored: np.ndarray, k: int) -> np.ndarray:
    '''
    Tournament selection: Selects the best individual from a random subset of the population.
    '''
    population_size = len(scored)
    winners = []

    for _ in range(population_size // 2):
        indices = np.random.choice(population_size, min(k, population_size), replace=False)
        selected = scored[indices]
        winner_idx = np.argmin(selected[:, 1])  # index of the lowest fitness
        winners.append(selected[winner_idx, 0])

    return np.array(winners, dtype=int)


def _rank_selection(scored: np.ndarray, k: int) -> np.ndarray:
    '''
    Rank selection: Assigns a rank to each individual based on their fitness and selects individuals based on their rank.
    '''
    population_size = len(scored)
    sorted_indices = np.argsort(scored[:, 1])
    ranks = np.arange(population_size, 0, -1)  # Higher rank = better

    probabilities = ranks / ranks.sum()
    selected_indices = np.random.choice(sorted_indices, size=population_size // 2, p=probabilities)
    return scored[selected_indices, 0].astype(int)


def _roulette_selection(scored: np.ndarray, k: int) -> np.ndarray:
    '''
    Roulette wheel selection: Selects individuals based on their fitness proportionate to the total fitness of the population.
    '''
    
    fitness = np.clip(scored[:, 1], 1e-6, None)  # avoid division by zero
    probabilities = (1 / fitness) / np.sum(1 / fitness)
    selected_indices = np.random.choice(len(scored), size=len(scored) // 2, p=probabilities)
    return scored[selected_indices, 0].astype(int)


def _top_half_selection(scored: np.ndarray, k: int) -> np.ndarray:
    '''
    Top half selection: Selects the top half of the population based on fitness.
    '''
    sorted_indices = np.argsort(scored[:, 1])
    top_half = sorted_indices[:len(scored)//2]
    return scored[top_half, 0].astype(int)


def _random_selection(scored: np.ndarray, k: int=None) -> np.ndarray:
    '''
    Random selection: Selects individuals randomly from the population.
    '''
    selected_indices = np.random.choice(len(scored), size=len(scored) // 2, replace=False)
    return scored[selected_indices, 0].astype(int)


class PARENT_SELECTION(enum.Enum):
    TOURNAMENT = "tournament"
    RANK = "rank"
    ROULETTE  = "roulette"
    TOP_HALF = "top_half"
    RANDOM = "random"


_parent_selection: Dict[PARENT_SELECTION, Callable[[List[List[int]], List[Tuple[List[int], float]], int], List[List[int]]]] = {
    PARENT_SELECTION.TOURNAMENT: _tournament_selection,
    PARENT_SELECTION.RANK: _rank_selection,
    PARENT_SELECTION.ROULETTE : _roulette_selection,
    PARENT_SELECTION.TOP_HALF: _top_half_selection,
    PARENT_SELECTION.RANDOM: _random_selection,
}


def select_parents( population: List[List[int]],
                    dist_matrix: ndarray,
                    route_distance_matrix:Callable[[Sequence[int], ndarray], float],
                    strategy: PARENT_SELECTION = PARENT_SELECTION.TOP_HALF,
                    k: int = 8, seed: Optional[int] = None
                ) -> List[List[int]]:
    """
    Selects parents from the population based on a given selection strategy.

    Args:
        population (List[List[int]]): 
            A list of candidate solutions (routes), where each route is a sequence of point indices.

        dist_matrix (ndarray):
            A precomputed distance matrix for the points.
        
        route_distance_matrix (Callable[[Sequence[int], ndarray], float]): 
            Function that computes the distance of a route using the distance matrix.

        strategy (PARENT_SELECTION): 
            The selection strategy to apply. Options are:
                - TOURNAMENT: Selects the best individual from a random subset of the population.
                - RANK: Selects individuals based on their fitness ranking.
                - ROULETTE: Selects individuals proportionally to their fitness.
                - TOP_HALF: Selects individuals from the top half of the population.
                - RANDOM: Selects individuals randomly.

        k (int): 
            Parameter used by some strategies (e.g., tournament size for TOURNAMENT).

    Returns:
        List[List[int]]: 
            A list of selected parent routes based on the specified strategy.
    """
    

    # Validate inputs
    if not population:
        raise ValueError("Population cannot be empty.")
    if k <= 0:
        raise ValueError("Parameter 'k' must be greater than 0.")

    if seed is not None:
        np.random.seed(seed)

    population_array = np.array(population, dtype=int)

    # Avalia as rotas
    scores = np.array([
        (i, route_distance_matrix(route.tolist(), dist_matrix)) for i, route in enumerate(population_array)
    ])
    
    selected_indices  = _parent_selection[strategy](scores, k)
    return [population_array[idx].tolist() for idx in selected_indices]

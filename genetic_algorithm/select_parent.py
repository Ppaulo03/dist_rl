from typing import List, Tuple, Dict, Callable, Optional
import random

import random
import enum


def _tournament_selection(population: List[List[int]], scored: List[Tuple[List[int], float]], k: int) -> List[List[int]]:
    '''
    Tournament selection: Selects the best individual from a random subset of the population.
    '''
    selected = []
    for _ in range(len(population) // 2):
        tournament = random.sample(scored, min(k, len(scored)))
        winner = min(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected


def _rank_selection(population: List[List[int]], scored: List[Tuple[List[int], float]], k: int) -> List[List[int]]:
    '''
    Rank selection: Assigns a rank to each individual based on their fitness and selects individuals based on their rank.
    '''
    sorted_scored = sorted(scored, key=lambda x: x[1])
    ranks = [len(sorted_scored) - i for i in range(len(sorted_scored))]
    selected = random.choices(sorted_scored, weights=ranks, k=len(population) // 2)
    return [route for route, _ in selected]


def _roulette_selection(population: List[List[int]], scored: List[Tuple[List[int], float]], k: int) -> List[List[int]]:
    '''
    Roulette wheel selection: Selects individuals based on their fitness proportionate to the total fitness of the population.
    '''
    safe_scores = [max(score, 1e-6) for _, score in scored]
    total_score = sum(1 / s for s in safe_scores)
    probabilities = [(1 / s) / total_score for s in safe_scores]
    selected = random.choices(scored, weights=probabilities, k=len(population) // 2)
    return [route for route, _ in selected]


def _top_half_selection(population: List[List[int]], scored: List[Tuple[List[int], float]], k: int) -> List[List[int]]:
    '''
    Top half selection: Selects the top half of the population based on fitness.
    '''
    sorted_scored = sorted(scored, key=lambda x: x[1])
    half_size = len(sorted_scored) // 2
    selected = [route for route, _ in sorted_scored[:half_size]]
    return selected


def _random_selection(population: List[List[int]], scored: List[Tuple[List[int], float]]=None, k: int=None) -> List[List[int]]:
    '''
    Random selection: Selects individuals randomly from the population.
    '''
    return random.sample(population, len(population) // 2)


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
                    points: List[Tuple[float, float]], 
                    origin: Tuple[float, float], 
                    distance_func: Callable[[Tuple[float, float], Tuple[float, float]], float],
                    strategy: PARENT_SELECTION = PARENT_SELECTION.TOP_HALF,
                    k: int = 8, seed: Optional[int] = None
                ) -> List[List[int]]:
    """
    Selects parents from the population based on a given selection strategy.

    Args:
        population (List[List[int]]): 
            A list of candidate solutions (routes), where each route is a sequence of point indices.

        points (List[Tuple[float, float]]): 
            Coordinates of all possible points in the problem space.

        origin (Tuple[float, float]): 
            Coordinates of the starting point.

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
        random.seed(seed)
    
    if strategy == PARENT_SELECTION.RANDOM: 
        return _random_selection(population)
    
    scored = [(route, distance_func(route, points, origin)) for route in population]
    return _parent_selection[strategy](population, scored, k)

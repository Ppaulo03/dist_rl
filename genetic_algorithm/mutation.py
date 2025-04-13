from typing import List, Dict, Callable, Optional
import random
import enum

def _swap_mutation(route: List[int], i: int, j: int):
    '''
    Swap mutation: Swaps two genes in the route.
    '''
    route[i], route[j] = route[j], route[i]


def _inversion_mutation(route: List[int], i: int, j: int):
    '''
    Inversion mutation: Reverses the order of a subsequence in the route.
    '''
    route[i:j] = reversed(route[i:j])


def _insertion_mutation(route: List[int], i: int, j: int):
    '''
    Insertion mutation: Moves a gene from one position to another in the route.
    '''
    item = route.pop(i)
    route.insert(j, item)


def _rotation_mutation(route: List[int], i: int, j: int):
    '''
    Rotation mutation: Rotates a subsequence in the route.
    '''
    if abs(i - j) > 1:
        route[i:j] = route[i:j][::-1]


def _non_adjacent_swap_mutation(route: List[int], i: int, j: int):
    '''
    Non-adjacent swap mutation: Swaps two non-adjacent genes in the route.
    '''
    if abs(i - j) > 1:
        route[i], route[j] = route[j], route[i]


def _scramble_mutation(route: List[int], i: int, j: int):
    '''
    Scramble mutation: Randomly shuffles a subsequence in the route.
    '''
    if abs(i - j) > 1:
        sub_route = route[i:j]
        random.shuffle(sub_route)
        route[i:j] = sub_route


def _circular_shift_mutation(route: List[int], i: int, j: int):
    '''
    Circular shift mutation: Shifts a subsequence in the route circularly.
    '''
    if abs(i - j) > 1:
        sub_route = route[i:j]
        route[i:j] = sub_route[-1:] + sub_route[:-1]
            

class MUTATION_SELECTION(enum.Enum):
    SWAP = "swap"
    INVERSION = "inversion"
    INSERTION = "insertion"
    ROTATION = "rotation"
    NON_ADJACENT_SWAP = "non_adjacent_swap"
    SCRAMBLE = "scramble"
    CIRCULAR_SHIFT = "circular_shift"


_mutation_selection: Dict[MUTATION_SELECTION, Callable[[List[int], int, int], None]] = {
    MUTATION_SELECTION.SWAP: _swap_mutation,
    MUTATION_SELECTION.INVERSION: _inversion_mutation,
    MUTATION_SELECTION.INSERTION: _insertion_mutation,
    MUTATION_SELECTION.ROTATION: _rotation_mutation,
    MUTATION_SELECTION.NON_ADJACENT_SWAP: _non_adjacent_swap_mutation,
    MUTATION_SELECTION.SCRAMBLE: _scramble_mutation,
    MUTATION_SELECTION.CIRCULAR_SHIFT: _circular_shift_mutation
}

_REQUIRES_NON_ADJACENT = {
    MUTATION_SELECTION.ROTATION,
    MUTATION_SELECTION.NON_ADJACENT_SWAP,
    MUTATION_SELECTION.SCRAMBLE,
    MUTATION_SELECTION.CIRCULAR_SHIFT
}


def mutate(route: List[int], mutation_rate: float, strategy: Optional[MUTATION_SELECTION]):
    '''
    Mutates the route based on the specified mutation rate and strategy.

    Args:
        route (List[int]): 
            The route to mutate, represented as a list of point indices.

        mutation_rate (float): 
            The probability of applying a mutation to the route.

        strategy (MUTATION_SELECTION, optional):
            The mutation strategy to use. If None, a random strategy will be selected. Options:
            - SWAP: Swaps the position of two cities in the route.
            - INVERSION: Reverses the order of a subsequence within the route.
            - INSERTION: Removes one city from the route and reinserts it at another position.
            - ROTATION: Rotates a subsequence of the route (similar to inversion but always applied).
            - NON_ADJACENT_SWAP: Swaps two cities only if they are not adjacent.
            - SCRAMBLE: Randomly shuffles the order of cities in a subsequence.
            - CIRCULAR_SHIFT: Performs a circular shift on a subsequence, moving elements by one position.

    Returns:
        List[int]: 
            The mutated route.
    '''

    if random.random() < mutation_rate:
        route = route.copy()
        if strategy is None:
            strategy = random.choice(list(MUTATION_SELECTION))

        if len(route) < 2:
            return route
        
        i, j = sorted(random.sample(range(len(route)), 2))
        if strategy in _REQUIRES_NON_ADJACENT:
            _timeout = 0
            while abs(i - j) <= 1:
                i, j = sorted(random.sample(range(len(route)), 2))
                _timeout += 1
                if _timeout > 10: break

        _mutation_selection[strategy](route, i, j)
    return route

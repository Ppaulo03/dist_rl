from typing import List, Dict, Callable, Optional
import random
import enum


def pmx_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Partially Mapped Crossover (PMX): Creates offspring by partially mapping genes from both parents.
    """
    size = len(parent1)
    child = [-1] * size

    # Randomly select two crossover points
    start, end = sorted(random.sample(range(size), 2))

    # Copy the segment from parent1 to child
    for i in range(start, end):
        child[i] = parent1[i]

    # Fill in the remaining genes from parent2
    for i in range(size):
        if child[i] == -1:
            gene = parent2[i]
            while gene in child:
                gene = parent2[parent1.index(gene)]
            child[i] = gene

    return child


def ox_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Order Crossover (OX): Creates offspring by preserving the order of genes from both parents.
    """
    size = len(parent1)
    child = [-1] * size

    # Randomly select two crossover points
    start, end = sorted(random.sample(range(size), 2))

    # Copy the segment from parent1 to child
    for i in range(start, end):
        child[i] = parent1[i]

    # Fill in the remaining genes from parent2 while preserving order
    p2_index = end % size
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_index] in child:
                p2_index = (p2_index + 1) % size
            child[i] = parent2[p2_index]
            p2_index = (p2_index + 1) % size

    return child


def cx_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Cycle Crossover (CX): Creates offspring by forming cycles between the two parents.
    """
    size = len(parent1)
    child = [-1] * size
    visited = [False] * size

    # Start with the first gene from parent1
    start = 0
    while -1 in child:
        if visited[start]:
            break
        current = start
        while True:
            child[current] = parent1[current]
            visited[current] = True
            current = parent2.index(parent1[current])
            if current == start:
                break

        # Move to the next unvisited gene in parent1
        start += 1
        while start < size and visited[start]:
            start += 1

    return child


def erx_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Edge Recombination Crossover (ERX): Creates offspring by preserving edges between genes.
    """
    size = len(parent1)
    child = [-1] * size
    edges = {i: set() for i in range(size)}

    # Build the edge list
    for i in range(size):
        edges[parent1[i]].add(parent2[i])
        edges[parent2[i]].add(parent1[i])

    # Start with a random gene from parent1
    current = random.choice(range(size))
    child[0] = current

    for i in range(1, size):
        next_gene = min(edges[current], key=lambda x: len(edges[x]))
        child[i] = next_gene
        edges[current].remove(next_gene)
        edges[next_gene].remove(current)
        current = next_gene

    return child


def pmxm_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Modified PMX Crossover (PMXM): A variation of PMX that uses a different mapping strategy.
    """
    size = len(parent1)
    child = [-1] * size

    # Randomly select two crossover points
    start, end = sorted(random.sample(range(size), 2))

    # Copy the segment from parent1 to child
    for i in range(start, end):
        child[i] = parent1[i]

    # Fill in the remaining genes from parent2 using a modified mapping strategy
    for i in range(size):
        if child[i] == -1:
            gene = parent2[i]
            while gene in child:
                gene = parent2[parent1.index(gene)]
            child[i] = gene

    return child


def pos_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Position-Based Crossover (POS): Creates offspring by selecting genes based on their positions in the parents.
    """
    size = len(parent1)
    child = [-1] * size

    # Randomly select a subset of genes from parent1
    selected_genes = random.sample(range(size), size // 2)

    # Copy the selected genes to child
    for i in selected_genes:
        child[i] = parent1[i]

    # Fill in the remaining genes from parent2 while preserving order
    p2_index = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_index] in child:
                p2_index += 1
            child[i] = parent2[p2_index]
            p2_index += 1

    return child


class CROSSOVER_STRATEGY(enum.Enum):
    PMX = "Partially Mapped Crossover"
    OX = "Order Crossover"
    CX = "Cycle Crossover"
    ERX = "Edge Recombination Crossover"
    PMXM = "Modified PMX"
    POS = "Position-Based Crossover"



_crossover_selection: Dict[CROSSOVER_STRATEGY, Callable[[List[int], List[int]], List[int]]] = {
    CROSSOVER_STRATEGY.PMX: pmx_crossover,
    CROSSOVER_STRATEGY.OX: ox_crossover,
    CROSSOVER_STRATEGY.CX: cx_crossover,
    CROSSOVER_STRATEGY.ERX: erx_crossover,
    CROSSOVER_STRATEGY.PMXM: pmxm_crossover,
    CROSSOVER_STRATEGY.POS: pos_crossover,
}


def crossover(parent1: List[int], parent2: List[int], strategy: CROSSOVER_STRATEGY, seed:Optional[int]=None) -> List[int]:
    """
    Performs crossover between two parents using the specified crossover strategy.

    Args:
        parent1 (List[int]): 
            The first parent.

        parent2 (List[int]): 
            The second parent.
            
        strategy (CROSSOVER_STRATEGY): 
            The crossover strategy to use. Options:
            - PMX: Partially Mapped Crossover, Creates offspring by partially mapping genes from both parents.
            - OX: Order Crossover, Creates offspring by preserving the order of genes from both parents.
            - CX: Cycle Crossover, Creates offspring by forming cycles between the two parents.
            - ERX: Edge Recombination Crossover, Creates offspring by preserving edges between genes.
            - PMXM: Modified PMX Crossover, A variation of PMX that uses a different mapping strategy.
            - POS: Position-Based Crossover, Creates offspring by selecting genes based on their positions in the parents.

        seed (int, optional): 
            Random seed for reproducibility.

    Returns:
        List[int]: 
            The offspring generated from the crossover.
    """

    if strategy not in _crossover_selection:
        raise ValueError(f"Invalid crossover strategy: {strategy}")

    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    if seed is not None:
        random.seed(seed)

    return _crossover_selection[strategy](parent1, parent2)


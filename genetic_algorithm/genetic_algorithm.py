try:
    from genetic_algorithm.haversine import total_distance
    from genetic_algorithm.select_parent import PARENT_SELECTION, select_parents
    from genetic_algorithm.mutation import MUTATION_SELECTION, mutate
except:
    from haversine import total_distance
    from select_parent import PARENT_SELECTION, select_parents
    from mutation import MUTATION_SELECTION, mutate

import random
import time


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


def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1]*size
    child[start:end] = parent1[start:end]

    fill = [p for p in parent2 if p not in child]
    ptr = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill[ptr]
            ptr += 1
    return child


def genetic_algorithm(points:list[tuple[float, float]], origin:tuple[float, float], 
                      generations:int=300, pop_size:int=150, mutation_rate:float=0.02, 
                      selection_method:PARENT_SELECTION=PARENT_SELECTION.RANK, k:int=5,
                      mutation_strategy:MUTATION_SELECTION=MUTATION_SELECTION.SWAP,
                      seeding:list[tuple[float, float]]=None, seeding_size:int=3, 
                      early_stop:int = 50, elitism:bool=True, verbose=False, use_threads:bool = False) -> list[tuple[float, float]]:
    num_points = len(points)
    population = create_population(pop_size, num_points, seeding, seeding_size)
    best_route = None
    best_distance = float('inf')

    no_improve_counter = 0
    for generation in range(generations):   

        # Seleciona os melhores (fitness)
        parents = select_parents(population, points, origin, selection_method, k=k)

        # Elitismo: guardar o melhor
        elite = min(parents, key=lambda r: total_distance(r, points, origin))
        elite_distance = total_distance(elite, points, origin)

        if elite_distance < best_distance:
            best_distance = elite_distance
            best_route = elite
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if early_stop > 0 and no_improve_counter >= early_stop:
            break #Exit early if no improvement for 50 generations
    
        # Preenche população com filhos
        next_population = []
        while len(next_population) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate, mutation_strategy)
            next_population.append(child)
        
        # Se elitismo estiver ativado, passa o elite pra próxima geração
        if elitism:
            next_population.append(elite)

        population = next_population

        # Print a cada 50 gerações
        if verbose and generation % 50 == 0:
            print(f"Geração {generation}: Melhor até agora = {best_distance:.2f} km")

    return best_route






if __name__ == "__main__":
    
    # Gerar pontos aleatórios para teste
    random.seed(42)  # Para reprodutibilidade
    points = [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(40)]
    origin = (0.0, 0.0)

    # for key in parent_selection.keys():
    #     start = time.time()
    #     best_route = genetic_algorithm(points, origin, selection_method=key)
    #     elapsed_time = time.time() - start
    #     dist = total_distance(best_route, points, origin)

    #     print('\nSelecionador:', key)
    #     print(f"Distância total: {dist:.2f} km")
    #     print(f"Tempo decorrido: {elapsed_time:.2f} segundos")


    random.seed(42)
    start = time.time()
    best_route = genetic_algorithm(points, origin, selection_method=PARENT_SELECTION.TOURNAMENT, k=8)
    elapsed_time = time.time() - start
    dist = total_distance(best_route, points, origin)

    print(f"Distância total: {dist:.2f} km")
    print(f"Tempo decorrido: {elapsed_time:.2f} segundos\n")


    random.seed(42)
    start = time.time()
    best_route = genetic_algorithm(points, origin, selection_method=PARENT_SELECTION.TOURNAMENT, k=8, use_threads=True)
    elapsed_time = time.time() - start
    dist = total_distance(best_route, points, origin)

    print(f"Distância total com Threads: {dist:.2f} km")
    print(f"Tempo decorrido com Threads: {elapsed_time:.2f} segundos\n")


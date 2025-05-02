from genetic_algorithm.routing_utils import nearest_neighbor, route_distance, two_opt
from genetic_algorithm.genetic_algorithm_model import genetic_algorithm
from genetic_algorithm.mutation import MUTATION_SELECTION
from genetic_algorithm.crossover import CROSSOVER_STRATEGY
from genetic_algorithm.select_parent import PARENT_SELECTION

import matplotlib.pyplot as plt
import random
import time
import csv

CROSSOVER = CROSSOVER_STRATEGY.OX
PARENT = PARENT_SELECTION.TOP_HALF
MUTATION = MUTATION_SELECTION.INVERSION

GA_POPSIZE = 125
GA_GENERATIONS = 300
GA_MUTATE_RATE = 0.02
EARLY_STOP = 50
VERBOSE = False

def time_complexity(min_points=10, max_points=200, step=10, pos_range_x=(-16.80, -16.50), pos_range_y=(-49.40, -49.10)):
    if max_points > 100 and min_points < 100:
        sizes_a = list(range(min_points, 101, step))
        sizes_b = list(range(200, max_points + 1, step*10))
        sizes = sizes_a + sizes_b
    else:
        sizes = list(range(min_points, max_points + 1, step))

    nn_times, opt_times, ga_times, nn_ga_times, ga_opt_times, nn_ga_opt_times = [], [], [], [], [], []
    nn_dists, opt_dists, ga_dists, nn_ga_dists, ga_opt_dists, nn_ga_opt_dists = [], [], [], [], [], []
    results = []
    origin = (0.0, 0.0)

    for n in sizes:
        # ----- Gerar pontos aleatórios
        random.seed(42 + n)
        points = [(random.uniform(*pos_range_x), random.uniform(*pos_range_y)) for _ in range(n)]

        pop_size = GA_POPSIZE
        generations = GA_GENERATIONS
        mutation_rate = GA_MUTATE_RATE
        early_stop = EARLY_STOP
        verbose = VERBOSE

        # ----- Nearest Neighbor -----
        start = time.time()
        nn_route = nearest_neighbor(origin, points)
        nn_time = time.time() - start
        nn_dist = route_distance(nn_route, points, origin)

        # ----- 2-Opt -----
        start = time.time()
        nn_route = nearest_neighbor(origin, points)
        opt_route = two_opt(nn_route, points, origin, max_iterations=10000)
        opt_time = time.time() - start
        opt_dist = route_distance(opt_route, points, origin)

        # ----- Genetic Algorithm -----
        start = time.time()
        ga_route = genetic_algorithm(points, origin, verbose=verbose, early_stop=early_stop, generations=generations, pop_size=pop_size, mutation_rate=mutation_rate, crossover_strategy=CROSSOVER, mutation_strategy=MUTATION, selection_method=PARENT)
        ga_time = time.time() - start
        ga_dist = route_distance(ga_route, points, origin)


        # # ----- NN + Genetic Algorithm -----
        # start = time.time()
        # nn_route = nearest_neighbor(origin, points)
        # nn_ga_route =  genetic_algorithm(points, origin, early_stop=early_stop, generations=generations, pop_size=pop_size, mutation_rate=mutation_rate, seeding=nn_route, crossover_strategy=CROSSOVER, mutation_strategy=MUTATION, selection_method=PARENT)
        # nn_ga_time = time.time() - start
        # nn_ga_dist = route_distance(nn_ga_route, points, origin)


        # # ----- Genetic Algorithm + 2-Opt -----
        # start = time.time()
        # ga_route = genetic_algorithm(points, origin, early_stop=early_stop, generations=generations, pop_size=pop_size, mutation_rate=mutation_rate, crossover_strategy=CROSSOVER, mutation_strategy=MUTATION, selection_method=PARENT)
        # ga_opt_route = two_opt(ga_route, points, origin, max_iterations=1000)
        # ga_opt_time = time.time() - start
        # ga_opt_dist = route_distance(ga_opt_route, points, origin)


        # # ----- NN + Genetic Algorithm + 2-Opt -----
        # start = time.time()
        # nn_route = nearest_neighbor(origin, points)
        # nn_ga_route =  genetic_algorithm(points, origin, early_stop=early_stop, generations=generations, pop_size=pop_size, mutation_rate=mutation_rate, seeding=nn_route, crossover_strategy=CROSSOVER, mutation_strategy=MUTATION, selection_method=PARENT)
        # nn_ga_opt_route = two_opt(nn_ga_route, points, origin, max_iterations=1000)
        # nn_ga_opt_time = time.time() - start
        # nn_ga_opt_dist = route_distance(nn_ga_opt_route, points, origin)


        # ----- Salvar resultados ----- 
        nn_times.append(nn_time)
        opt_times.append(opt_time)
        ga_times.append(ga_time)
        #nn_ga_times.append(nn_ga_time)
        #ga_opt_times.append(ga_opt_time)
        #nn_ga_opt_times.append(nn_ga_opt_time)

        nn_dists.append(nn_dist)
        opt_dists.append(opt_dist)
        ga_dists.append(ga_dist)
        #nn_ga_dists.append(nn_ga_dist)
        #ga_opt_dists.append(ga_opt_dist)
        #nn_ga_opt_dists.append(nn_ga_opt_dist)

        results.append({
            "Pontos": n,
            "NN_Dist": round(nn_dist, 2), "NN_Time": round(nn_time, 4), 
            "2Opt_Dist": round(opt_dist, 2), "2Opt_Time": round(opt_time, 4),
            "GA_Dist": round(ga_dist, 2), "GA_Time": round(ga_time, 4),
            #"NN_GA_Dist": round(nn_ga_dist, 2), "NN_GA_Time": round(nn_ga_time, 4),
            #"GA_2Opt_Dist": round(ga_opt_dist, 2), "GA_2Opt_Time": round(ga_opt_time, 4),
            #"NN_GA_2Opt_Dist": round(nn_ga_opt_dist, 2), "NN_GA_2Opt_Time": round(nn_ga_opt_time, 4),
        })

        # ----- Log ----- 
        print(f"[{n} pontos] NN: {nn_dist:.1f}km | 2-Opt: {opt_dist:.1f}km | GA: {ga_dist:.1f}km")
        print(f"Tempo → NN: {nn_time:.4f}s | 2-Opt: {opt_time:.4f}s | GA: {ga_time:.4f}s")
        print("-" * 120)



    # ----- Salvar em CSV -----
    fieldnames = [
        "Pontos",
        "NN_Dist", "2Opt_Dist", "GA_Dist", "NN_GA_Dist", "GA_2Opt_Dist", "NN_GA_2Opt_Dist",
        "NN_Time", "2Opt_Time", "GA_Time", "NN_GA_Time", "GA_2Opt_Time", "NN_GA_2Opt_Time"
    ]

    with open("data/resultados_tsp.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ----- Plot -----
    _, axs = plt.subplots(2, 1, figsize=(12, 6))

    axs[0].plot(sizes, nn_times, label='Nearest Neighbor', marker='o', color='green')
    axs[0].plot(sizes, opt_times, label='2-Opt', marker='o', color='blue')
    axs[0].plot(sizes, ga_times, label='Genetic Algorithm', marker='o', color='orange')
    #axs[0].plot(sizes, nn_ga_times, label='NN + Genetic Algorithm', marker='o', color='purple')
    #axs[0].plot(sizes, ga_opt_times, label='GA + 2-Opt', marker='o', color='red')
    #axs[0].plot(sizes, nn_ga_opt_times, label='NN + GA + 2-Opt', marker='o', color='black')
    axs[0].set_yscale('log')  # Escala logarítmica para melhor visualização

    axs[0].set_title('Tempo de Computação x Número de Pontos')
    axs[0].set_xlabel('Número de Pontos')
    axs[0].set_ylabel('Tempo (segundos)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sizes, nn_dists, label='Nearest Neighbor', marker='o', color='green')
    axs[1].plot(sizes, opt_dists, label='2-Opt', marker='o', color='blue')
    axs[1].plot(sizes, ga_dists, label='Genetic Algorithm', marker='o', color='orange')
    #axs[1].plot(sizes, nn_ga_dists, label='NN + Genetic Algorithm', marker='o', color='purple')
    #axs[1].plot(sizes, ga_opt_dists, label='GA + 2-Opt', marker='o', color='red')
    #axs[1].plot(sizes, nn_ga_opt_dists, label='NN + GA + 2-Opt', marker='o', color='black')
    axs[1].set_title('Distância x Número de Pontos')
    axs[1].set_xlabel('Número de Pontos')
    axs[1].set_ylabel('Distância (km)')
    axs[1].legend()
    axs[1].grid(True)

    plt.subplots_adjust(hspace=0.4)
    plt.show()




if __name__ == "__main__":
    time_complexity(max_points=100)

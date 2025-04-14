from genetic_algorithm.routing_utils import nearest_neighbor, route_distance, two_opt
from genetic_algorithm.genetic_algorithm_model import genetic_algorithm
from genetic_algorithm.mutation import _mutation_selection
from genetic_algorithm.crossover import _crossover_selection
from genetic_algorithm.select_parent import _parent_selection

import matplotlib.pyplot as plt
import random
import time
import csv

def compare_routes(num_points=20, pos_range_x=(-50, 50), pos_range_y=(-50, 50)):
    def plot_route(ax, points, origin, route, title):
        path = [origin] + [points[i] for i in route] + [origin]
        xs, ys = zip(*path)
        ax.plot(xs, ys, marker='o', linestyle='-')
        ax.set_title(title)
        ax.scatter(*zip(*points), color='red')
        ax.scatter(origin[0], origin[1], color='green', s=100, label='Origem')
        ax.legend()

    # === Executar Experimento ===
    points = [(random.uniform(pos_range_x[0], pos_range_x[1]), random.uniform(pos_range_y[0], pos_range_y[1])) for _ in range(num_points)]
    origin = (0.0, 0.0)
    point_indices = list(range(len(points)))

    # Nearest Neighbor
    start = time.time()
    nn_route = nearest_neighbor(origin, points)
    nn_dist = route_distance(nn_route, points, origin)
    nn_time = time.time() - start
    print(f"Distância Nearest Neighbor: {nn_dist:.2f} km | Tempo: {nn_time:.2f} s")

    # 2-Opt
    start = time.time()
    opt_route = two_opt(nn_route, points, origin)
    opt_dist = route_distance(opt_route, points, origin)
    opt_time = time.time() - start
    print(f"Distância 2-Opt: {opt_dist:.2f} km | Tempo: {opt_time:.2f} s")

    # Genetic Algorithm
    start = time.time()
    ga_route = genetic_algorithm(points, origin, generations=500, pop_size=200, mutation_rate=0.05)
    ga_dist = route_distance(ga_route, points, origin)
    ga_time = time.time() - start
    print(f"Distância Genetic Algorithm: {ga_dist:.2f} km | Tempo: {ga_time:.2f} s")

    # NN + Genetic Algorithm
    nn_ga_route = genetic_algorithm(points, origin, generations=400, pop_size=200, mutation_rate=0.05, seeding=nn_route)
    nn_ga_dist = route_distance(nn_ga_route, points, origin)
    print(f"Distância NN + Genetic Algorithm: {nn_ga_dist:.2f} km")

    # Genetic Algorithm + 2-Opt
    ga_opt_route = two_opt(ga_route, points, origin, max_iterations=1000)
    ga_opt_dist = route_distance(ga_opt_route, points, origin)
    print(f"Distância Genetic Algorithm + 2-Opt: {ga_opt_dist:.2f} km")

    # NN + Genetic Algorithm + 2-Opt
    nn_ga_opt_route = two_opt(nn_ga_route, points, origin, max_iterations=1000)
    nn_ga_opt_dist = route_distance(nn_ga_opt_route, points, origin)
    print(f"Distância NN + Genetic Algorithm + 2-Opt: {nn_ga_opt_dist:.2f} km")

 
    # Plot comparativo
    fig, axs = plt.subplots(2, 3, figsize=(18, 6))

    plot_route(axs[0][0], points, origin, nn_route, f'Nearest Neighbor\nDist: {nn_dist:.2f} km')
    plot_route(axs[0][1], points, origin, opt_route, f'2-Opt\nDist: {opt_dist:.2f} km')
    plot_route(axs[0][2], points, origin, ga_route, f'Genetic Algorithm\nDist: {ga_dist:.2f} km')
    plot_route(axs[1][0], points, origin, nn_ga_route, f'NN + Genetic Algorithm\nDist: {nn_ga_dist:.2f} km')
    plot_route(axs[1][1], points, origin, ga_opt_route, f'GA + 2-Opt\nDist: {ga_opt_dist:.2f} km')
    plot_route(axs[1][2], points, origin, nn_ga_opt_route, f'NN + GA + 2-Opt\nDist: {nn_ga_opt_dist:.2f} km')

    plt.suptitle('Comparação de Rotas: NN x 2-Opt x GA', fontsize=16)
    plt.subplots_adjust(hspace=0.4)  # Ajusta o espaço entre os subplots
    plt.show()


def time_complexity(min_points=10, max_points=200, step=10, pos_range_x=(-50, 50), pos_range_y=(-50, 50)):
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
        point_indices = list(range(n))

        pop_size = min(300, max(50, n // 2))
        generations = min(600, max(100, n * 4))

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
        ga_route = genetic_algorithm(points, origin, generations=generations, pop_size=pop_size, mutation_rate=0.05)
        ga_time = time.time() - start
        ga_dist = route_distance(ga_route, points, origin)


        # ----- NN + Genetic Algorithm -----
        start = time.time()
        nn_route = nearest_neighbor(origin, points)
        nn_ga_route =  genetic_algorithm(points, origin, generations=generations, pop_size=pop_size, mutation_rate=0.05, seeding=nn_route)
        nn_ga_time = time.time() - start
        nn_ga_dist = route_distance(nn_ga_route, points, origin)


        # ----- Genetic Algorithm + 2-Opt -----
        start = time.time()
        ga_route = genetic_algorithm(points, origin, generations=generations, pop_size=pop_size, mutation_rate=0.05)
        ga_opt_route = two_opt(ga_route, points, origin, max_iterations=1000)
        ga_opt_time = time.time() - start
        ga_opt_dist = route_distance(ga_opt_route, points, origin)


        # ----- NN + Genetic Algorithm + 2-Opt -----
        start = time.time()
        nn_route = nearest_neighbor(origin, points)
        nn_ga_route =  genetic_algorithm(points, origin, generations=generations, pop_size=pop_size, mutation_rate=0.05, seeding=nn_route)
        nn_ga_opt_route = two_opt(nn_ga_route, points, origin, max_iterations=1000)
        nn_ga_opt_time = time.time() - start
        nn_ga_opt_dist = route_distance(nn_ga_opt_route, points, origin)


        # ----- Salvar resultados ----- 
        nn_times.append(nn_time)
        opt_times.append(opt_time)
        ga_times.append(ga_time)
        nn_ga_times.append(nn_ga_time)
        ga_opt_times.append(ga_opt_time)
        nn_ga_opt_times.append(nn_ga_opt_time)

        nn_dists.append(nn_dist)
        opt_dists.append(opt_dist)
        ga_dists.append(ga_dist)
        nn_ga_dists.append(nn_ga_dist)
        ga_opt_dists.append(ga_opt_dist)
        nn_ga_opt_dists.append(nn_ga_opt_dist)

        results.append({
            "Pontos": n,
            "NN_Dist": round(nn_dist, 2), "NN_Time": round(nn_time, 4), 
            "2Opt_Dist": round(opt_dist, 2), "2Opt_Time": round(opt_time, 4),
            "GA_Dist": round(ga_dist, 2), "GA_Time": round(ga_time, 4),
            "NN_GA_Dist": round(nn_ga_dist, 2), "NN_GA_Time": round(nn_ga_time, 4),
            "GA_2Opt_Dist": round(ga_opt_dist, 2), "GA_2Opt_Time": round(ga_opt_time, 4),
            "NN_GA_2Opt_Dist": round(nn_ga_opt_dist, 2), "NN_GA_2Opt_Time": round(nn_ga_opt_time, 4),
        })

        # ----- Log ----- 
        print(f"[{n} pontos] NN: {nn_dist:.1f}km | 2-Opt: {opt_dist:.1f}km | GA: {ga_dist:.1f}km | NN+GA: {nn_ga_dist:.1f}km | GA+2-Opt: {ga_opt_dist:.1f}km | NN+GA+2-Opt: {nn_ga_opt_dist:.1f}km")
        print(f"Tempo → NN: {nn_time:.4f}s | 2-Opt: {opt_time:.4f}s | GA: {ga_time:.4f}s | NN+GA: {nn_ga_time:.4f}s | GA+2-Opt: {ga_opt_time:.4f}s | NN+GA+2-Opt: {nn_ga_opt_time:.4f}s")
        print("-" * 120)



    # ----- Salvar em CSV -----
    fieldnames = [
        "Pontos",
        "NN_Dist", "2Opt_Dist", "GA_Dist", "NN_GA_Dist", "GA_2Opt_Dist", "NN_GA_2Opt_Dist",
        "NN_Time", "2Opt_Time", "GA_Time", "NN_GA_Time", "GA_2Opt_Time", "NN_GA_2Opt_Time"
    ]

    with open("resultados_tsp.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


    # ----- Plot -----
    _, axs = plt.subplots(2, 1, figsize=(12, 6))

    axs[0].plot(sizes, nn_times, label='Nearest Neighbor', marker='o', color='green')
    axs[0].plot(sizes, opt_times, label='2-Opt', marker='o', color='blue')
    axs[0].plot(sizes, ga_times, label='Genetic Algorithm', marker='o', color='orange')
    axs[0].plot(sizes, nn_ga_times, label='NN + Genetic Algorithm', marker='o', color='purple')
    axs[0].plot(sizes, ga_opt_times, label='GA + 2-Opt', marker='o', color='red')
    axs[0].plot(sizes, nn_ga_opt_times, label='NN + GA + 2-Opt', marker='o', color='black')
    axs[0].set_yscale('log')  # Escala logarítmica para melhor visualização

    axs[0].set_title('Tempo de Computação x Número de Pontos')
    axs[0].set_xlabel('Número de Pontos')
    axs[0].set_ylabel('Tempo (segundos)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sizes, nn_dists, label='Nearest Neighbor', marker='o', color='green')
    axs[1].plot(sizes, opt_dists, label='2-Opt', marker='o', color='blue')
    axs[1].plot(sizes, ga_dists, label='Genetic Algorithm', marker='o', color='orange')
    axs[1].plot(sizes, nn_ga_dists, label='NN + Genetic Algorithm', marker='o', color='purple')
    axs[1].plot(sizes, ga_opt_dists, label='GA + 2-Opt', marker='o', color='red')
    axs[1].plot(sizes, nn_ga_opt_dists, label='NN + GA + 2-Opt', marker='o', color='black')
    axs[1].set_title('Distância x Número de Pontos')
    axs[1].set_xlabel('Número de Pontos')
    axs[1].set_ylabel('Distância (km)')
    axs[1].legend()
    axs[1].grid(True)

    plt.subplots_adjust(hspace=0.4)
    plt.show()


def log(n, results, strats):
    log_output_dist = f"[{n} pontos]" + " | ".join([f"{strat}: {results[strat][-1][1]:.1f}km" for strat in strats])
    log_output_time = f"[{n} pontos]" + " | ".join([f"{strat}: {results[strat][-1][2]:.4f}s" for strat in strats])
    print(log_output_dist)
    print(log_output_time)
    print("-" * 120)


def save_csv(results, strats, filename):
    fieldnames = ["Pontos", "Estrategia", "Distância", "Tempo"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for strat in strats:
            for n, dist, comp_time in results[strat]:
                writer.writerow({"Pontos": n, "Estrategia": strat, "Distância": dist, "Tempo": comp_time})


def plot_results(results, strats):
    _, axs = plt.subplots(2, 1, figsize=(12, 6))

    for strat in strats:
        axs[0].plot([x[0] for x in results[strat]], [x[1] for x in results[strat]], label=strat, marker='o')
        axs[1].plot([x[0] for x in results[strat]], [x[2] for x in results[strat]], label=strat, marker='o')
    
    axs[0].set_yscale('log')  # Escala logarítmica para melhor visualização
    axs[0].set_title('Tempo de Computação x Número de Pontos')
    axs[0].set_xlabel('Número de Pontos')
    axs[0].set_ylabel('Tempo (segundos)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Distância x Número de Pontos')
    axs[1].set_xlabel('Número de Pontos')
    axs[1].set_ylabel('Distância (km)')
    axs[1].legend()
    axs[1].grid(True)

    plt.subplots_adjust(hspace=0.4)
    plt.show()



def compare_mutations_strategy(min_points=10, max_points=200, step=10, pos_range_x=(-50, 50), pos_range_y=(-50, 50), plot=True):
    strats = _mutation_selection.keys()
    results = {k: [] for k in strats}

    if max_points > 100 and min_points < 100:
        sizes_a = list(range(min_points, 101, step))
        sizes_b = list(range(200, max_points + 1, step*10))
        sizes = sizes_a + sizes_b
    else:
        sizes = list(range(min_points, max_points + 1, step))

    for n in sizes:
        random.seed(42 + n)
        points = [(random.uniform(*pos_range_x), random.uniform(*pos_range_y)) for _ in range(n)]
        origin = (0.0, 0.0)

        for strat in strats:
            start = time.time()
            ga_route = genetic_algorithm(points, origin, mutation_rate=0.05, mutation_strategy=strat)
            ga_time = time.time() - start
            ga_dist = route_distance(ga_route, points, origin)

            results[strat].append((n, ga_dist, ga_time))
        log(n, results, strats)
        
    
    # ----- Salvar em CSV -----
    save_csv(results, strats, "resultados_crossover.csv")
        
    # ----- Plot -----
    if plot:
        plot_results(results, strats)


def compare_select_parent_strategy(min_points=10, max_points=200, step=10, pos_range_x=(-50, 50), pos_range_y=(-50, 50), plot=True):
    strats = _parent_selection.keys()
    results = {k: [] for k in strats}

    if max_points > 100 and min_points < 100:
        sizes_a = list(range(min_points, 101, step))
        sizes_b = list(range(200, max_points + 1, step*10))
        sizes = sizes_a + sizes_b
    else:
        sizes = list(range(min_points, max_points + 1, step))
    
    for n in sizes:
        random.seed(42 + n)
        points = [(random.uniform(*pos_range_x), random.uniform(*pos_range_y)) for _ in range(n)]
        origin = (0.0, 0.0)

        for strat in strats:
            start = time.time()
            ga_route = genetic_algorithm(points, origin, selection_method=strat)
            ga_time = time.time() - start
            ga_dist = route_distance(ga_route, points, origin)

            results[strat].append((n, ga_dist, ga_time))
        log(n, results, strats)
        
    
    # ----- Salvar em CSV -----
    save_csv(results, strats, "resultados_select_parents.csv")
        
    # ----- Plot -----
    if plot:
        plot_results(results, strats)


def compare_crossover_strategy(min_points=10, max_points=200, step=10, pos_range_x=(-50, 50), pos_range_y=(-50, 50), plot=True):
    strats = _crossover_selection.keys()
    results = {k: [] for k in strats}

    if max_points > 100 and min_points < 100:
        sizes_a = list(range(min_points, 101, step))
        sizes_b = list(range(200, max_points + 1, step*10))
        sizes = sizes_a + sizes_b
    else:
        sizes = list(range(min_points, max_points + 1, step))
    
    for n in sizes:
        random.seed(42 + n)
        points = [(random.uniform(*pos_range_x), random.uniform(*pos_range_y)) for _ in range(n)]
        origin = (0.0, 0.0)

        for strat in strats:
            start = time.time()
            ga_route = genetic_algorithm(points, origin, crossover_strategy=strat)
            ga_time = time.time() - start
            ga_dist = route_distance(ga_route, points, origin)

            results[strat].append((n, ga_dist, ga_time))
        log(n, results, strats)
        
    
    # ----- Salvar em CSV -----
    save_csv(results, strats, "resultados_mutacoes.csv")
        
    # ----- Plot -----
    if plot:
        plot_results(results, strats)
    

if __name__ == "__main__":
    #compare_routes(num_points=50)
    #time_complexity(max_points=100)
    compare_mutations_strategy(min_points=10, max_points=100, step=10, plot=False)
    compare_select_parent_strategy(min_points=10, max_points=100, step=10, plot=False)
    compare_crossover_strategy(min_points=10, max_points=100, step=10)
    pass

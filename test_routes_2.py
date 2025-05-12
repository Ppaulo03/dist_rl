from gen2.model import genetic_algorithm
from gen2.utils import nearest_neighbor, route_distance, two_opt
import numpy as np
import osmnx as ox

import matplotlib.pyplot as plt
import time

GA_POPSIZE = 1000
GA_GENERATIONS = 30000
GA_MUTATE_RATE = 0.2

ORIGIN = (-49.238858, -16.671274)
place = "Goiânia, Goiás, Brasil"
G = ox.graph_from_place(place, network_type='drive')
G_proj = ox.project_graph(G)
G_proj = G_proj.to_undirected()

class Result:
    def __init__(self, name, name_short, route, distance, time):
        self.name = name
        self.name_short = name_short
        self.route = route
        self.distance = distance
        self.time = time

    def __str__(self):
        return f"{self.name}: {self.distance:.2f} km, {self.time:.4f} s"


def run_all(origin, points, verbose=False):
    #Nearest Neighbor
    start = time.time()
    nn_route, dist_matrix = nearest_neighbor(origin, points)
    nn_dist = route_distance(nn_route, dist_matrix)
    nn_time = time.time() - start
    nn_result = Result("Nearest Neighbor", "NN", nn_route, nn_dist, nn_time)
    if verbose: print(nn_result)

    # 2-Opt
    start = time.time()
    nn_route, dist_matrix = nearest_neighbor(origin, points)
    opt_route = two_opt(nn_route, dist_matrix)
    opt_dist = route_distance(opt_route, dist_matrix)
    opt_time = time.time() - start
    opt_result = Result("2-Opt", "2-Opt", opt_route, opt_dist, opt_time)
    if verbose: print(opt_result)

    # Genetic Algorithm
    start = time.time()
    ga_route, ga_dist = genetic_algorithm(points, origin, seeding=False, post_process=False, population_size=GA_POPSIZE, generations=GA_GENERATIONS, mutation_rate=GA_MUTATE_RATE)
    ga_time = time.time() - start
    ga_result = Result("Genetic Algorithm", "GA", ga_route, ga_dist, ga_time)
    if verbose: print(ga_result)

    # NN + Genetic Algorithm
    start = time.time()
    nn_ga_route, nn_ga_dist = genetic_algorithm(points, origin, seeding=True, post_process=False, population_size=GA_POPSIZE, generations=GA_GENERATIONS, mutation_rate=GA_MUTATE_RATE)
    nn_ga_time = time.time() - start
    nn_ga_result = Result("NN + Genetic Algorithm", "NN+GA", nn_ga_route, nn_ga_dist, nn_ga_time)
    if verbose: print(nn_ga_result)

    # Genetic Algorithm + 2-Opt
    start = time.time()
    ga_opt_route, ga_opt_dist = genetic_algorithm(points, origin, seeding=False, post_process=True, population_size=GA_POPSIZE, generations=GA_GENERATIONS, mutation_rate=GA_MUTATE_RATE)
    ga_opt_time = time.time() - start
    ga_opt_result = Result("GA + 2-Opt", "GA+2-Opt", ga_opt_route, ga_opt_dist, ga_opt_time)
    if verbose: print(ga_opt_result)

    # NN + Genetic Algorithm + 2-Opt
    start = time.time()
    nn_ga_opt_route, nn_ga_opt_dist = genetic_algorithm(points, origin, seeding=True, post_process=True, population_size=GA_POPSIZE, generations=GA_GENERATIONS, mutation_rate=GA_MUTATE_RATE)
    nn_ga_opt_time = time.time() - start
    nn_ga_opt_result = Result("NN + GA + 2-Opt", "NN+GA+2-Opt", nn_ga_opt_route, nn_ga_opt_dist, nn_ga_opt_time)
    if verbose: print(nn_ga_opt_result)

    return {
        "NN": nn_result,
        "2-Opt": opt_result,
        "GA": ga_result,
        "NN + GA": nn_ga_result,
        "GA + 2-Opt": ga_opt_result,
        "NN + GA + 2-Opt": nn_ga_opt_result
    }
    

def compare_routes(num_points=20, origin=ORIGIN, plot=True):
    def plot_route(ax, points, origin, route, title):
        path = [origin] + [points[i] for i in route] + [origin]
        xs, ys = zip(*path)
        ax.plot(xs, ys, marker='o', linestyle='-')
        ax.set_title(title)
        ax.scatter(*zip(*points), color='red')
        ax.scatter(origin[0], origin[1], color='green', s=100, label='Origem')
        ax.legend()

    points_proj = ox.utils_geo.sample_points(G_proj, num_points)
    points_geo = points_proj.to_crs(epsg=4326)
    points = [[point.x, point.y] for point in points_geo.geometry]
    points = np.array(points)

    results = run_all(origin, points, verbose=True)

    if not plot: return results
    # Plot comparativo
    fig, axs = plt.subplots(2, 3, figsize=(18, 6))
    for i, result in enumerate(results.values()):
        plot_route(axs[i//3][i%3], points, origin, result.route, f'{result.name}\nDist: {result.distance:.2f} km')

    plt.suptitle('Comparação de Rotas: NN x 2-Opt x GA', fontsize=16)
    plt.subplots_adjust(hspace=0.4)  # Ajusta o espaço entre os subplots
    plt.show()


def time_complexity(min_points=10, max_points=200, step=10, origin=ORIGIN):
    
    if max_points > 100 and min_points < 100:
        sizes_a = list(range(min_points, 101, step))
        sizes_b = list(range(200, max_points + 1, step*10))
        sizes = sizes_a + sizes_b
    else:
        sizes = list(range(min_points, max_points + 1, step))

    # ----- Inicializar listas para armazenar os resultados ----
    nn_results, opt_results, ga_results = [], [], []
    nn_ga_results, ga_opt_results, nn_ga_opt_results = [], [], []

    for n in sizes:
        # ----- Gerar pontos aleatórios
        points_proj = ox.utils_geo.sample_points(G_proj, n)
        points_geo = points_proj.to_crs(epsg=4326)
        points = [[point.x, point.y] for point in points_geo.geometry]
        points = np.array(points)

   
        results = run_all(origin, points)

        # ----- Salvar resultados ----- 
        nn_results.append(results["NN"])
        opt_results.append(results["2-Opt"])
        ga_results.append(results["GA"])
        nn_ga_results.append(results["NN + GA"])
        ga_opt_results.append(results["GA + 2-Opt"])
        nn_ga_opt_results.append(results["NN + GA + 2-Opt"])


        # ----- Log ----- 
        log_dist = f"[{n} pontos] "
        log_time = "Tempo → "
        for result in results.values():
            log_dist += f"{result.name_short}: {result.distance:.1f}km | "
            log_time += f"{result.name_short}: {result.time:.4f}s | "
        print(log_dist[:-3])  # Remove o último " | "
        print(log_time[:-3])  # Remove o último " | "
        print("-" * 120)


    # ----- Plot -----
    _, axs = plt.subplots(2, 1, figsize=(12, 6))

    nn_times = [result.time for result in nn_results]
    opt_times = [result.time for result in opt_results]
    ga_times = [result.time for result in ga_results]
    nn_ga_times = [result.time for result in nn_ga_results]
    ga_opt_times = [result.time for result in ga_opt_results]
    nn_ga_opt_times = [result.time for result in nn_ga_opt_results]
    nn_dists = [result.distance for result in nn_results]
    opt_dists = [result.distance for result in opt_results]
    ga_dists = [result.distance for result in ga_results]
    nn_ga_dists = [result.distance for result in nn_ga_results]
    ga_opt_dists = [result.distance for result in ga_opt_results]
    nn_ga_opt_dists = [result.distance for result in nn_ga_opt_results]

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

    

if __name__ == "__main__":
    time_complexity(max_points=1000)
    #compare_routes(num_points=20)
    

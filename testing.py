from genetic_algorithm.routing_utils import  route_distance, nearest_neighbor
from geneticalgorithm import geneticalgorithm as ga
import numpy as np
import random
import time
LAT = -16.6869
LON = -49.2648

# Variação em torno de Goiânia (em graus)
LAT_RANGE = 0.06
LON_RANGE = 0.06
ORIGIN = (-16.6864, -49.4000)
N_POINTS = 50

pos_range_x=(LAT - LAT_RANGE, LAT + LAT_RANGE)
pos_range_y=(LON - LON_RANGE, LON + LON_RANGE)
points = [(random.uniform(*pos_range_x), random.uniform(*pos_range_y)) for _ in range(N_POINTS)]

def f(X):
    route = X.astype(int).tolist()
    distance = route_distance(route, points, ORIGIN)
    
    return distance
start = time.time()
model = ga(
            function=f, 
            dimension=N_POINTS, 
            variable_type='int', 
            variable_boundaries=np.array([[0, N_POINTS-1]]*N_POINTS), 
        )

model.run()
elapsed_time = time.time() - start
print(f"Tempo de execução: {elapsed_time:.2f} segundos")
print("Melhor percurso encontrado:")
print(model.output_dict['variable'])
print("Distância total do percurso:")
print(model.output_dict['function'])

start = time.time()
nn_route = nearest_neighbor(ORIGIN, points)
elapsed_time = time.time() - start
print(f"Tempo de execução do algoritmo vizinho mais próximo: {elapsed_time:.2f} segundos")
print("Melhor percurso encontrado com o algoritmo vizinho mais próximo:")
print(nn_route)
print("Distância total do percurso com o algoritmo vizinho mais próximo:")
print(route_distance(nn_route, points, ORIGIN))

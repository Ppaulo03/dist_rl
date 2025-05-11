from gen2.utils import nearest_neighbor as nn1
from gen2.utils import route_distance as rd1
from gen2.utils import two_opt as two_opt1

from genetic_algorithm.routing_utils import nearest_neighbor as nn2
from genetic_algorithm.routing_utils import route_distance as rd2
from genetic_algorithm.routing_utils import two_opt as two_opt2
import numpy as np
import time

ORIGIN = (-16.6864, -49.4000)
LAT = -16.6869
LAT_RANGE = 0.06
LON = -49.2648
LON_RANGE = 0.06
N_POINTS = 100

lat_min, lat_max = LAT - LAT_RANGE, LAT + LAT_RANGE
lon_min, lon_max = LON - LON_RANGE, LON + LON_RANGE

lats = np.random.uniform(lat_min, lat_max, size=N_POINTS)
lons = np.random.uniform(lon_min, lon_max, size=N_POINTS)

points = np.column_stack((lats, lons))

from gen2.model import genetic_algorithm

print('\n\n-----')
start = time.time()
ga_route = genetic_algorithm(points, ORIGIN, population_size=300, generations=1000, mutation_rate=0.2)
dist_ga = rd2(ga_route, points, ORIGIN)
elapsed_time_ga = time.time() - start
print("Distance GA:", dist_ga)
print("Elapsed time for genetic algorithm:", elapsed_time_ga)
print("-----")


start = time.time()
nn_route, dist_matrix = nn1(ORIGIN, points)
dist_nn = rd1(nn_route, dist_matrix)
elapsed_time_nn = time.time() - start
print("Distance NN:", dist_nn)
print("Elapsed time for NN:", elapsed_time_nn)
print("-----")


start = time.time()
two_opt_route = two_opt1(nn_route, dist_matrix)
dist_two_opt = rd1(two_opt_route, dist_matrix)
elapsed_time_two_opt = time.time() - start + elapsed_time_nn
print("Distance 2-opt:", dist_two_opt)
print("Elapsed time for 2-opt:", elapsed_time_two_opt)






# points = points.tolist()
# start = time.time()
# nn_route2 = nn2(ORIGIN, points)
# opt2 = two_opt2(nn_route2, points, ORIGIN)
# dist_2 = rd2(opt2, points, ORIGIN)
# elapsed_time2 = time.time() - start
# print("Distance 2:", dist_2)
# print("Elapsed time for algorithm2:", elapsed_time2)

# print("-----")
# print('Diff between distances:', dist_2 - dist_1, 'km')
# print('Diff between elapsed times:', elapsed_time2 - elapsed_time1)
# print('Efficiency gain:', (elapsed_time2 - elapsed_time1) / elapsed_time2 * 100, '%')


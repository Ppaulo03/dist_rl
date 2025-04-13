from genetic_algorithm.haversine import haversine, total_distance


def two_opt(route, points, origin, max_iterations=-1):
    best = route
    best_distance = total_distance(best, points, origin)
    iteration = 0
    improved = True

    while improved and (max_iterations < 0 or iteration < max_iterations):
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue  # vizinhos, pular
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                new_distance = total_distance(new_route, points, origin)

                if new_distance < best_distance:
                    best = new_route
                    best_distance = new_distance
                    improved = True

        iteration += 1

    return best

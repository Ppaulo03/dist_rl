from genetic_algorithm.haversine import haversine

def nearest_neighbor(origin, points, point_indices):
    if not point_indices: return []

    unvisited = set(point_indices)
    current_pos = origin
    route = []

    while unvisited:
        nearest = min(unvisited, key=lambda idx: haversine(current_pos, points[idx]))
        route.append(nearest)
        current_pos = points[nearest]
        unvisited.remove(nearest)

    return route
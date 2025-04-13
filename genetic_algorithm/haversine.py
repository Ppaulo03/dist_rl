import math


R = 6371 # Earth radius in m
class Coords():
    def __init__(self, latitude: float, longitude: float):
        self.latitude = math.radians(latitude)
        self.longitude = math.radians(longitude)
    

def haversine(c1:tuple[float, float], c2: tuple[float, float]) -> float: 
    c1 = Coords(c1[0], c1[1])
    c2 = Coords(c2[0], c2[1])   
    delta_phi = c2.latitude - c1.latitude
    delta_lambda = c2.longitude - c1.longitude

    a = math.sin(delta_phi/2)**2 + math.cos(c1.latitude) * math.cos(c2.latitude) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def total_distance(route, points, origin):
    path = [origin] + [points[i] for i in route] + [origin]
    return sum(haversine(path[i], path[i+1]) for i in range(len(path)-1))
import osmnx as ox
import matplotlib.pyplot as plt

# Define a cidade de interesse
place = "Goiânia, Goiás, Brasil"

# Baixa a rede viária para veículos motorizados
G = ox.graph_from_place(place, network_type='drive')

# Projeta o grafo para um sistema de coordenadas adequado
G_proj = ox.project_graph(G)

# Gera 20 pontos aleatórios ao longo da rede viária
points = ox.utils_geo.sample_points(G_proj, 20)

points_geo = points.to_crs(epsg=4326)
locations = [[point.x, point.y] for point in points_geo.geometry]
print(locations)
from env import RouteOptimizationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from haversine import haversine

# Callback para salvar modelo periodicamente
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path='./checkpoints/',
    name_prefix='ppo_route_model'
)

# Carregar pontos (latitude, longitude)
points = []
with open("goiania_cords.csv", "r") as file:
    file.readline()  # Pula o cabeçalho
    for line in file:
        lat, lon = map(float, line.strip().split(","))
        points.append((lat, lon))

# Criar ambiente
env = RouteOptimizationEnv(points)


# === Função Nearest Neighbor ===
def nearest_neighbor(origin, points, point_indices):
    if not point_indices:
        return []

    unvisited = set(point_indices)
    current_pos = origin
    route = []

    while unvisited:
        nearest = min(unvisited, key=lambda idx: haversine(current_pos, points[idx]))
        route.append(nearest)
        current_pos = points[nearest]
        unvisited.remove(nearest)

    return route

# Treinar PPO
policy_kwargs = dict(
    net_arch=[64, 32],
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cuda")
model.learn(total_timesteps=100_000_000, callback=checkpoint_callback)

# === Testando ===


n_episodes = 5  # Número de episódios que você quer rodar

for episode in range(n_episodes):
    obs, _ = env.reset()  # Reseta o ambiente para o início de cada episódio
    total_reward = 0  # Variável para armazenar a recompensa total do episódio
    total_ep = 0  # Contador de passos do episódio
    for _ in range(1000):  # Limite de passos por episódio
        action, _ = model.predict(obs)

        # Filtra pontos selecionados pelo agente
        selected_points = [i for i in range(len(action)) if action[i] == 1]

        # Ordena com Nearest Neighbor
        ordered_route = nearest_neighbor(env.env.origin, env.env.points, selected_points)
        # Aplica a rota ordenada no ambiente
        obs, reward, done, _, _ = env.step(ordered_route)
        total_reward += reward  # Soma a recompensa total do episódio
        total_ep += 1
 
        if done:
            print(f"Episódio {episode+1} finalizado com recompensa total média: {total_reward/total_ep}")
            break
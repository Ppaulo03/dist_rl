import numpy as np
import random
from typing import List, Tuple
from haversine import haversine

import gymnasium as gym
from gymnasium import spaces

ORIGIN = (-16.626614, -49.267072)  # Origin coordinates
OVERFLOW_THRESHOLD = 90  # Percentage threshold for overflow penalty
MIN_RESET_LEVEL = 70  # Minimum level to reset a point
MAX_INITIAL_LEVEL = 60  # Maximum initial level for points
REWARDS = {
    'correct_reset': 100,
    'early_reset': 0,
    'distance_penalty': 0,
    'overflow_penalty': 0,
}


class _RouteOptimizationEnv:
    def __init__(self, points: List[Tuple[float, float]], level_increase_range=(1, 50)):
        self.origin = ORIGIN
        self.points = points
        self.num_points = len(points)
        self.levels = np.zeros(self.num_points)
        self.level_increase_range = level_increase_range
        self.reset()


    def reset(self):
        self.agent_position = self.origin
        self.levels = np.random.uniform(0, MAX_INITIAL_LEVEL, self.num_points)  # Random start levels below 50%
        return self._get_state()


    def _get_state(self):
        return np.array(self.levels)


    def step(self, route: List[int]):
        total_distance = 0
        reward = 0

        current_pos = self.origin

        # Calculate distance and reward for each point in the route
        for idx, point_idx in enumerate(route):
            point = self.points[point_idx]
            distance = self._compute_distance(current_pos, point)
            total_distance += distance
            current_pos = point

            
            if self.levels[point_idx] < MIN_RESET_LEVEL: reward += REWARDS['early_reset']  # Penalty for resetting too early
            else: reward += REWARDS['correct_reset']  # Reward for correct reset
            self.levels[point_idx] = 0

        
        # Return to origin
        total_distance += self._compute_distance(current_pos, self.origin)
        reward += total_distance * REWARDS['distance_penalty']  # Distance penalty


        # Levels rise randomly after full route
        for i in range(self.num_points):
            increase = random.uniform(*self.level_increase_range)
            self.levels[i] = min(100, self.levels[i] + increase)

        # Heavy penalty if any point went over 90%
        reward += sum(REWARDS['overflow_penalty'] for lvl in self.levels if lvl >= OVERFLOW_THRESHOLD)
        reward /= 1000  # Normalize reward

        done = any(lvl >= OVERFLOW_THRESHOLD for lvl in self.levels)
        return self._get_state(), reward, done
    

    def _compute_distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return haversine(a, b)



class RouteOptimizationEnv(gym.Env):
    def __init__(self, points):
        super(RouteOptimizationEnv, self).__init__()
        self.env = _RouteOptimizationEnv(points)  # Seu ambiente customizado
        self.action_space = spaces.MultiDiscrete([2] * len(points))  # Máscara binária
        self.observation_space = spaces.Box(low=0, high=100, shape=(len(points),), dtype=np.float32)

    def reset(self, seed=None):
        state = self.env.reset()
        return state, {} 

    def step(self, action):
        state, reward, done = self.env.step(action)
        terminated = done
        truncated = False  # Se você não tem um critério de "truncamento", pode deixar False.
        return state, reward, terminated, truncated, {}
    

if __name__ == '__main__':

    # Carregar pontos (latitude, longitude)
    points = []
    with open("goiania_cords.csv", "r") as file:
        file.readline()  # Pula o cabeçalho
        for line in file:
            lat, lon = map(float, line.strip().split(","))
            points.append((lat, lon))

    # Criar ambiente
    env = RouteOptimizationEnv(points)

    # Teste rápido do ambiente
    obs, _ = env.reset()
    action = [1] * len(points)  # Seleciona todos os pontos
    new_obs, reward, done, _, _ = env.step(action)

    print(f"\n\nObservação inicial: {obs}")
    print(f"Ação: {action}")
    print(f"Nova observação: {new_obs}")
    print(f"Recompensa: {reward}")
    print(f"Finalizado: {done}")
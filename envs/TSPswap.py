import gym
import random
from envs.TSP import TSP

class TSPSwap(TSP):
    def __init__(self, num_cities=5):
        self.num_cities = num_cities

    def initial_problem(self):
        tour = self._generate_problem()

        return tour

    def step(self, state, action):
        distance = self._tour_distance(state)
        a = state[action[0]]
        b = state[action[1]]

        state[action[0]] = b
        state[action[1]] = a

        new_distance = self._tour_distance(state)
        reward = self._reward(distance, new_distance)

        return state, reward, False

    def _reward(self, dist, new_dist):
        return dist - new_dist








if __name__ == '__main__':
    env = TSPSwap()
    print(env._generate_problem())

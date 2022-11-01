import math
import random
import gym
import numpy as np
import torch

class TSP:
    def __init__(self, num_cities=5):
        self.num_cities = num_cities

    def initial_problem(self):
        state = {'unscheduled': self._generate_problem(), 'tour': []}

        return state

    def step(self, state, action):
        reward, done = 0, False
        unscheduled, tour = state['unscheduled'][:], state['tour'][:]
        tour.append(unscheduled.pop(action))

        if len(unscheduled) == 0:
            reward, done = self._reward(tour), True

        return {'unscheduled': unscheduled, 'tour': tour}, reward, done

    def _tour_distance(self, tour):
        distance = 0
        for i, city in enumerate(tour):
            if i < len(tour) - 1:
                distance += self._euclidean_distance(city, tour[i + 1])

        return distance

    def _reward(self, tour):
        distance = self._tour_distance(tour)

        return -distance

    def _euclidean_distance(self, a, b):
        return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

    def legal_actions(self, state):
        return [i for i in range(len(state['unscheduled']))]

    def _generate_problem(self):
        cities = []

        for i in range(self.num_cities):
            cities.append(self._generate_coords())

        return cities

    def _generate_coords(self):
        x = random.random()
        y = random.random()

        return x, y

    def create_obs(self, state):
        unscheduled = state['unscheduled'][:]
        tour = state['tour'][:]
        unscheduled.extend([(-1, -1)] * (self.num_cities - len(unscheduled)))
        tour.extend([(-1, -1)] * (self.num_cities - len(tour)))

        obs = unscheduled + tour

        obs = [item for t in obs for item in t]

        return obs



# class TSPGym(gym.Env):
#     def __init__(self, num_cities=5):
#         self.num_cities = num_cities
#         self.env = TSP(num_cities)
#         self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[4*num_cities], dtype=np.float32)
#         self.action_space = gym.spaces.Discrete(num_cities)
#
#     def reset(self):
#         self.state = self.env.initial_problem()
#         return self.create_obs(self.state)
#
#     def step(self, action):
#         legal = self.env.legal_actions(self.state)
#         if action in legal:
#             self.state, reward, done = self.env.step(self.state, action)
#         else:
#             reward = -1
#             done = False
#
#         return self.create_obs(self.state), reward, done, dict()
#
#     def create_obs(self, state):
#         unscheduled = state['unscheduled'][:]
#         tour = state['tour'][:]
#         unscheduled.extend([(-1, -1)] * (self.num_cities - len(unscheduled)))
#         tour.extend([(-1, -1)] * (self.num_cities - len(tour)))
#
#         obs = unscheduled + tour
#
#         obs = [item for t in obs for item in t]
#
#         return obs
#
#     def render(self):
#         from envs.tour_plotter import plot_state
#         plot_state(self.state['tour'])
#

class TSPGym(gym.Env, TSP):
    def __init__(self, num_cities=5):
        self.num_cities = num_cities
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[4*num_cities], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(num_cities)

        self.reset()

    def reset(self):
        self.state = self.initial_problem()
        return self.create_obs(self.state)

    def raw_state(self):
        return self.state

    def step(self, action):
        legal = self.legal_actions(self.state)
        if action in legal:
            self.state, reward, done = TSP.step(self, self.state, action)
        else:
            reward = -1
            done = False

        return self.create_obs(self.state), reward, done, dict()


    def render(self):
        from envs.tour_plotter import plot_state
        plot_state(self.state['tour'])




def stb3_policy_probs(model, obs, num_actions):
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    return torch.exp(model.policy.evaluate_actions(obs_tensor, torch.tensor([i for i in range(num_actions)]))[1])

if __name__ == '__main__':
    # env = TSP()
    #
    # state = env.initial_problem()
    # print(state)
    # done = False
    #
    # while not done:
    #     legal_actions = env.legal_actions(state)
    #     action = random.choice(legal_actions)
    #     state, reward, done = env.step(state, action)
    #     print(state, reward)


    env = TSPGym(num_cities=15)

    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="stb3_tsp_tensorboard/")
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_tsp.zip")
    model = PPO.load("ppo_tsp")

    obs = env.reset()
    done = False
    while not done:

        print(stb3_policy_probs(model, obs, 155))

        # print(model.policy.evaluate_actions())
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    env.render()
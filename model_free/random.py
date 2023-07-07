from pathlib import Path
import random
import math
import numpy as np

from experiment_management.config_handling.load_exp_config import load_yml
from experiment_management.setup_experiment import create_env

env_config = load_yml((Path('data/config/envs/jsp_minimal_003.yml')))
env, _, _ = create_env(env_config)


class RandomAgent:
    def __init__(self, env):
        self.env = env
    def predict(self, obs, **kwargs):
        legal_actions = self.env.model.legal_actions(self.env.raw_state())
        return random.choice(legal_actions), None


class SPTAgent:
    def __init__(self, env):
        self.env = env

    def predict(self, obs, **kwargs):
        legal_actions = self.env.model.legal_actions(self.env.raw_state())

        min_duration = math.inf
        action = -1
        for a in legal_actions:
            duration = self.env.raw_state()['remaining_operations'][a][0].duration
            if duration < min_duration:
                action = a
                min_duration = duration

        return action, None

agent = RandomAgent(env)
obs = env.reset()
done = False

rewards = []
episodes = 1000
for _ in range(episodes):
    obs = env.reset()
    done = False
    reward = 0
    while not done:
        action, _state = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    rewards.append(reward)

print(np.mean(rewards))

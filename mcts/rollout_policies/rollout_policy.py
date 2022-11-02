import random


class RolloutPolicy:
    def __init__(self, model):
        self.model = model

    def rollout(self, state):
        raise NotImplementedError


class RandomRolloutPolicy(RolloutPolicy):
    def __init__(self, model):
        self.model = model

    def rollout(self, state):
        done = False

        while not done:
            legal_actions = self.model.legal_actions(state)
            action = random.choice(legal_actions)
            state, reward, done = self.model.step(state, action)

        return reward

    def __str__(self):
        return "RandRoll"
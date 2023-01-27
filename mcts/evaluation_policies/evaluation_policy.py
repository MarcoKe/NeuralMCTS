import random


class EvaluationPolicy:
    def evaluate(self, state, **kwargs):
        raise NotImplementedError


class RandomRolloutPolicy(EvaluationPolicy):
    def evaluate(self, state, model=None, **kwargs):
        done = False

        while not done:
            legal_actions = model.legal_actions(state)
            action = random.choice(legal_actions)
            state, reward, done = model.step(state, action)

        return reward

    def __str__(self):
        return "RandRoll"
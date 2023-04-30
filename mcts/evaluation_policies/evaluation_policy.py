import copy
import random
import torch


class EvaluationPolicy:
    def evaluate_multiple(self, states, **kwargs):
        """
        When using neural nets, it can be more efficient to evaluate multiple states at the same time
        """
        values = []
        priors = []
        for s in states:
            value, priors_ = self.evaluate(copy.deepcopy(s), **kwargs)
            values.append(value)
            if torch.is_tensor(priors_) or priors_:
                priors.append(priors_)

        if len(priors) == 0:
            priors = None

        return values, priors

    def evaluate(self, state, **kwargs):
        raise NotImplementedError


class RandomRolloutPolicy(EvaluationPolicy):
    def evaluate(self, state, model=None, env=None, **kwargs):
        done = False

        while not done:
            legal_actions = model.legal_actions(state)

            action = random.choice(legal_actions)
            state, reward, done = model.step(state, action)
            reward = env.reward(reward)

        return reward, None

    def __str__(self):
        return "RandRoll"
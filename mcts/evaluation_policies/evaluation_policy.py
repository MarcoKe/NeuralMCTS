import copy
import random
from typing import List

import torch

from mcts.node import Node


class EvaluationPolicy:
    def evaluate_multiple(self, nodes, states, **kwargs):
        """
        When using neural nets, it can be more efficient to evaluate multiple states at the same time
        """
        values = []
        priors = []
        trajectories = []
        for s in states:
            value, priors_, trajectory = self.evaluate(copy.deepcopy(s), **kwargs)
            values.append(value)
            if torch.is_tensor(priors_) or priors_:
                priors.append(priors_)
            trajectories.append(trajectory)

        if len(priors) == 0:
            priors = None

        return values, priors, trajectories

    def evaluate(self, node, state, **kwargs):
        raise NotImplementedError


class RandomRolloutPolicy(EvaluationPolicy):
    def evaluate(self, node: Node, state, model=None, env=None, **kwargs):
        done = False
        trajectory: List[(Node, )] = [(node, state)]

        while not done:
            # choose action
            legal_actions = model.legal_actions(state)
            action = random.choice(legal_actions)

            # add to trajectory
            trajectory.append((Node(action, trajectory[-1][0]), copy.deepcopy(state)))

            # step
            state, reward, done = model.step(state, action)
            reward = env.reward(reward)

        return reward, None, trajectory

    def __str__(self):
        return "RandomRollout"
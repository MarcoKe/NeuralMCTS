import time

from mcts.tree_policies.tree_policy import UCTPolicy
from mcts.node import Node
import torch
import random


class NeuralPUCTPolicy(UCTPolicy):
    # "Predictor Upper Confidence bounds applied to Trees"
    def __init__(self, exploration_const, agent, model):
        self.exp_const = exploration_const
        self.agent = agent
        self.model = model

    def stb3_policy_probs(self, obs, actions):  # used in U function
        obs_tensor = self.agent.policy.obs_to_tensor(self.model.create_obs(obs))[0]
        value, logprob, entropy = self.agent.policy.evaluate_actions(obs_tensor, torch.tensor(actions))
        return torch.exp(logprob)  # likelihood of taking each action

    def stb3_value(self, obs):  # used in Q function
        obs_tensor = self.agent.policy.obs_to_tensor(self.model.create_obs(obs))[0]
        value, logprob, entropy = self.agent.policy.evaluate_actions(obs_tensor, torch.tensor([1]))
        return value  # of the observation

    def U(self, node, child, state):
        return self.exp_const * child.action_prob * node.visits / (0 + child.visits)  # todo: check whether there is a difference between node visits and the summed visits of all children

    def select(self, node: Node, state):
        best_uct, child = -1e7, None
        for c in node.children:
            # if c.visits == 0:
            #     s_, _, _ = self.model.step(state, c.action)
            #     c.returns = self.stb3_value(s_)
            #     c.visits = 1

            if c.visits > 0 and node.visits > 0:
                uct = self.Q(c) + self.U(node, c, state)
                if uct > best_uct:
                    best_uct = uct
                    child = c

        if not child:
            child = random.choice(node.children)

        return child

    def __str__(self):
        return "NPUCT"
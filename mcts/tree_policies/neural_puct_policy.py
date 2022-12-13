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

    def stb3_policy_probs(self, obs, actions):
        obs_tensor = self.agent.policy.obs_to_tensor(self.model.create_obs(obs))[0]
        value, logprob, entropy = self.agent.policy.evaluate_actions(obs_tensor, torch.tensor(actions))
        return torch.exp(logprob)

    def stb3_value(self, obs):
        obs_tensor = self.agent.policy.obs_to_tensor(self.model.create_obs(obs))[0]
        # obs_tensor_list = [obs_tensor for _ in range(1000)]
        # obs_tensor_batch = torch.cat(obs_tensor_list, 0)
        value, logprob, entropy = self.agent.policy.evaluate_actions(obs_tensor, torch.tensor([1]))
        return value

    def U(self, node, child, state):
        return self.exp_const * self.stb3_policy_probs(state, child.action) * node.visits / (0 + child.visits) # todo: check whether there is a difference between node visits and the summed visits of all children

    def expand(self, node: Node, state):
        if node.visits == 0:  # initialize the node's return value
            s_, _, _ = self.model.step(state, node.action)
            node.returns = self.stb3_value(s_)  # used in the UCTPolicy's Q function
            node.visits = 1
        else:  # check if node.returns already initialized if the node has been visited
            print(node.returns)

    def select(self, node: Node, state):
        best_uct, child = -1e7, None
        start = time.time()
        for c in node.children:
            self.expand(c, state)

            # if c.visits > 0 and node.visits > 0:
            uct = self.Q(c) + self.U(node, c, state)
            if uct > best_uct:
                best_uct = uct
                child = c

        # returns = []
        # for c in node.children:
        #     if c.visits == 0:
        #         s_, _, _ = self.model.step(state, c.action)
        #         returns.append(s_)
        #         c.visits = 1
        #
        # s = self.stb3_value(returns)
        # for i, c in enumerate(node.children):
        #     c.returns = s[i]
        #     uct = self.Q(c) + self.U(node, c, state)
        #     if uct > best_uct:
        #         best_uct = uct
        #         child = c
        #
        # print("Average child returns calculation time: " + str((time.time() - start) / len(node.children))
        #       if len(node.children) > 0 else "-")

        if not child:
            child = random.choice(node.children)

        return child

    def __str__(self):
        return "NPUCT"
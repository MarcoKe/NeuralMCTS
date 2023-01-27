import copy
import math
import torch
import numpy as np
from typing import List
from mcts.tree_policies.tree_policy import TreePolicy
from mcts.evaluation_policies.evaluation_policy import EvaluationPolicy
from mcts.expansion_policies.expansion_policy import ExpansionPolicy
from mcts.node import Node
from model_free.stb3_wrapper import RLAgent


class MCTSAgent:
    def __init__(self, model, tree_policy: TreePolicy, expansion_policy: ExpansionPolicy,
                 evaluation_policy: EvaluationPolicy, neural_net: RLAgent,
                 num_simulations=10, dirichlet_noise=False):
        self.model = model
        self.tree_policy = tree_policy
        self.expansion_policy = expansion_policy
        self.evaluation_policy = evaluation_policy
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.dirichlet_noise = dirichlet_noise

    def init_tree(self, n, s):
        """
        create a path from the root of the tree to a terminal node by greedily choosing nodes with the learned policy
        this should make the results of a tree search at least as good as the learned policy without tree search would have been
        but only under the assumption that value estimates are perfect
        """
        done = False

        while not done:
            new_children = self.expansion_policy.expand(n, s, model=self.model, neural_net=self.neural_net)
            state_value, action_probs = self.evaluation_policy.evaluate(s, model=self.model, neural_net=self.neural_net)
            # children_states = [self.model.step(state, c.action)[0] for c in new_children]
            children_state_values = self.neural_net.state_values(
                [self.model.create_obs(self.model.step(s, c.action)[0]) for c in new_children])

            max_prior = -math.inf
            best_child = None
            for i, c in enumerate(new_children):
                c.prior_prob = action_probs[i]
                c.update(
                    children_state_values[i][0])  # maybe create a class to define how new children are initialized?
                if c.prior_prob > max_prior:
                    max_prior = c.prior_prob
                    best_child = c

            n = best_child
            s, terminal_reward, done = self.model.step(s, n.action)

        while n.has_parent():
            n.update(terminal_reward)
            n = n.parent
        n.update(state_value)

    def mcts_search(self, state, mode='mean'):
        root_node = Node(None, None)

        self.init_tree(root_node, copy.deepcopy(state))

        simulation_count = 0
        while simulation_count < self.num_simulations:
            simulation_count += 1

            n, s = root_node, copy.deepcopy(state)

            done = False
            while not n.is_leaf():
                n = self.tree_policy.select(n, add_dirichlet=(n.is_root() and self.dirichlet_noise))
                s, terminal_reward, done = self.model.step(s, n.action)

            if not done:
                new_children = self.expansion_policy.expand(n, s, model=self.model, neural_net=self.neural_net)
                state_value, action_probs = self.evaluation_policy.evaluate(s, model=self.model, neural_net=self.neural_net)
                # children_states = [self.model.step(state, c.action)[0] for c in new_children]
                children_state_values = self.neural_net.state_values([self.model.create_obs(self.model.step(s, c.action)[0]) for c in new_children])
                for i, c in enumerate(new_children):
                    c.prior_prob = action_probs[i]
                    c.update(children_state_values[i][0]) # maybe create a class to define how new children are initialized?

            else:
                state_value = terminal_reward

            while n.has_parent():
                n.update(state_value)
                n = n.parent
            n.update(state_value)

        return root_node

    def select_action(self, state, mode='max'):
        root_node = self.mcts_search(state, mode)
        action, value = root_node.select_best_action(mode)

        return action, value

    @staticmethod
    def exponentiated_visit_counts(counts: List[int], total: int, temperature: float) -> np.array:
        exponentiated_visit_counts = np.array(counts) / total
        exponentiated_visit_counts = np.power(exponentiated_visit_counts, temperature)
        return exponentiated_visit_counts / sum(exponentiated_visit_counts)

    def stochastic_policy(self, state, temperature: float = 0.9):
        root_node = self.mcts_search(state)
        visit_counts = [0] * self.model.max_num_actions()
        for c in root_node.children:
            visit_counts[c.action] = c.visits

        policy = self.exponentiated_visit_counts(visit_counts, root_node.visits, temperature)
        value = root_node.returns / root_node.visits
        return policy, value, root_node.select_best_action()[0]

    def __str__(self):
        return "MCTS(" + str(self.tree_policy) + ", " + str(self.evaluation_policy) + ")" + str(self.dirichlet_noise)


if __name__ == '__main__':
    from envs.TSP import TSPGym, TSP
    env = TSPGym(num_cities=15)
    model = TSP(num_cities=15)


    from tree_policies.tree_policy import UCTPolicy
    from tree_policies.exploration_terms.puct_term import PUCTTerm
    from tree_policies.exploitation_terms.avg_node_value import AvgNodeValueTerm
    from evaluation_policies.neural_value_eval import NeuralValueEvalPolicy
    from stable_baselines3 import PPO
    from model_free.stb3_wrapper import Stb3ACAgent

    model_free_agent = PPO.load("results/trained_agents/tsp/model_free/ppo_tsp_15_3e6")
    tp = UCTPolicy(AvgNodeValueTerm(), PUCTTerm(exploration_constant=1))
    ep = ExpansionPolicy()
    rp = NeuralValueEvalPolicy()
    agent = MCTSAgent(model, tp, ep, rp, neural_net=Stb3ACAgent(model_free_agent), num_simulations=1000, dirichlet_noise=True)

    num_iter = 100

    rewards = 0
    for _ in range(num_iter):
        state = env.reset()
        done = False

        while not done:
            action, _ = agent.select_action(env.raw_state())
            state, reward, done, _ = env.step(action)

        rewards += reward

        env.render()
    print("avg rew: ", rewards / num_iter)
    #
    #
    # rewards = 0
    # for _ in range(num_iter):
    #     state = env.reset()
    #     done = False
    #
    #     while not done:
    #         action, _ = model_free_agent.predict(state, deterministic=True)
    #         state, reward, done, _ = env.step(action)
    #
    #     rewards += reward
    #
    #     env.render()
    # print("avg rew: ", rewards / num_iter)
    #





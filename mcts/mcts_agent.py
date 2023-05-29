import copy
import math
import numpy as np
from typing import List
from random import choices

import torch

from mcts.tree_policies.tree_policy import TreePolicy
from mcts.evaluation_policies.evaluation_policy import EvaluationPolicy
from mcts.expansion_policies.expansion_policy import ExpansionPolicy
from mcts.node import Node
from model_free.stb3_wrapper import RLAgent, EvalCounterWrapper
from envs.model import ModelStepCounter


class MCTSAgent:
    def __init__(self, env, model, tree_policy: TreePolicy, expansion_policy: ExpansionPolicy,
                 evaluation_policy: EvaluationPolicy, neural_net: RLAgent,
                 num_simulations=10, dirichlet_noise=False, evaluate_leaf_children=False, value_initialization=True,
                 initialize_tree=True, **kwargs):
        self.env = env
        self.model = model
        self.tree_policy = tree_policy
        self.expansion_policy = expansion_policy
        self.evaluation_policy = evaluation_policy
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.dirichlet_noise = dirichlet_noise
        self.evaluate_leaf_children = evaluate_leaf_children   # if False, apply evaluation policy to encountered leaf. If True, apply evaluation policy to expanded children of leaf
        self.value_initialization = value_initialization  # this only matters if evaluate_leaf_children is False
        self.initialize_tree = initialize_tree  # whether to populate tree with greedy neural net rollout

    def init_prior(self, action_probs, new_children, s):
        if torch.is_tensor(action_probs) or action_probs:
            for i, c in enumerate(new_children):
                c.prior_prob = action_probs[i]
        else:
            self.tree_policy.exploration_term.init_prior(new_children, state=copy.deepcopy(s), env=self.env,
                                                         neural_net=self.neural_net)

    def init_tree(self, n, s):
        """
        create a path from the root of the tree to a terminal node by greedily choosing nodes with the learned policy
        this should make the results of a tree search at least as good as the learned policy without tree search would have been
        but only under the assumption that value estimates are perfect
        """
        done = False

        while not done:
            new_children = self.expansion_policy.expand(n, s, model=self.model, env=self.env, neural_net=self.neural_net)
            state_value, action_probs = self.evaluation_policy.evaluate(s, model=self.model, env=self.env, neural_net=self.neural_net)
            # children_states = [self.model.step(state, c.action)[0] for c in new_children]
            children_state_values = self.neural_net.state_values(
                [self.env.observation(self.model.step(s, c.action)[0]) for c in new_children])

            self.init_prior(action_probs, new_children, s)
            max_prior = -math.inf
            best_child = None
            for i, c in enumerate(new_children):
                c.update(
                    children_state_values[i][0])  # maybe create a class to define how new children are initialized?

                if not c.prior_prob:  # this is ugly: if there are no prior probs, we just use the values instead
                    if c.value() > max_prior:
                        max_prior = c.value()
                        best_child = c

                elif c.prior_prob > max_prior:
                    max_prior = c.prior_prob
                    best_child = c

            n = best_child
            s, terminal_reward, done = self.model.step(s, n.action)
            terminal_reward = self.env.reward(terminal_reward)

        while n.has_parent():
            n.update(terminal_reward)
            n = n.parent
        n.update(terminal_reward)

    def mcts_search(self, state, mode='mean'):
        model = ModelStepCounter(self.model)
        neural_net = EvalCounterWrapper(self.neural_net)
        root_node = Node(None, None)

        if self.initialize_tree: self.init_tree(root_node, copy.deepcopy(state))

        simulation_count = 0
        while simulation_count < self.num_simulations:
            simulation_count += 1

            n, s = root_node, copy.deepcopy(state)

            n, s, terminal_reward, done = self.selection_phase(n, s, model)

            if not done:
                new_children = self.expansion_phase(n, s, model, neural_net)
                self.evaluation_phase(n, s, new_children, model, neural_net)

            else:
                state_value = terminal_reward
                self.backpropagation_phase(n, state_value)

        stats = {'model_steps': model.count, 'neural_net_calls': neural_net.count}
        return root_node, stats

    def selection_phase(self, n, s, model):
        done = False
        terminal_reward = None
        while not n.is_leaf():
            n = self.tree_policy.select(n, add_dirichlet=(n.is_root() and self.dirichlet_noise))
            s, terminal_reward, done = model.step(s, n.action) #todo: check if model reward should go through reward fun wrapper
            terminal_reward = self.env.reward(terminal_reward)
        return n, s, terminal_reward, done

    def expansion_phase(self, n, s, model, neural_net):
        new_children = self.expansion_policy.expand(n, s, model=model, env=self.env, neural_net=neural_net)
        return new_children

    def evaluation_phase(self, n, s, new_children, model, neural_net):
        """
        Two distinct modes depending on self.evaluate_leaf_children:
        if True, the evaluation policy is applied to evaluate every newly expanded child of the encountered leaf node
        if False, the evaluation policy is only applied to the encountered lead node. In this case, the values of the
        expanded children still need to be initialized somehow. This can either be +inf, to force exploration of
        unvisited nodes, or initialization by the learned value function to give a reasonable estimate.
        Initialization by value function only affects the value of the child. The value is not backpropagated up the tree.
        This is in contrast to evaluation of all leaf children with a learned value function.
        """
        if self.evaluate_leaf_children:
            state_values = []
            states = []
            for c in new_children:
                s_, reward_, done_ = model.step(copy.deepcopy(s), c.action)
                reward_ = self.env.reward(reward_)
                states.append(s_)

                # with this strategy, we need to check if the episodes are done at this point. otherwise, we will pass
                # a terminal state to the evaluation policy
                if done_:
                    state_values.append(reward_)

            if len(state_values) == 0:
                state_values, action_probs = self.evaluation_policy.evaluate_multiple(copy.deepcopy(states), model=model, env=self.env,
                                                                                  neural_net=neural_net)

            for i, value in enumerate(state_values):
                child = new_children[i]
                self.backpropagation_phase(child, value)

                # initialize action priors if the evaluation policy computed them automatically.
                # if torch.is_tensor(action_probs) or action_probs:
                #     child.prior_prob = action_probs[i]
                # else:  # otherwise compute them only if the tree policy requires them
                self.tree_policy.exploration_term.init_prior(new_children, state=copy.deepcopy(s), env=self.env,
                                                             neural_net=neural_net)


        else:
            state_value, action_probs = self.evaluation_policy.evaluate(copy.deepcopy(s), model=model, env=self.env,
                                                                        neural_net=neural_net)
            self.backpropagation_phase(n, state_value)

            self.init_prior(action_probs, new_children, s)

            if self.value_initialization:
                children_state_values = neural_net.state_values(
                    [self.env.observation(model.step(copy.deepcopy(s), c.action)[0]) for c in new_children])
                children_state_values = [c[0] for c in children_state_values]
            else:
                children_state_values = [math.inf for _ in new_children]

            for i, c in enumerate(new_children):
                c.update(children_state_values[i])

    def backpropagation_phase(self, n, value):
        while n.has_parent():
            n.update(value)
            n = n.parent
        n.update(value)

    def select_action(self, state, mode='mean'):
        root_node, _ = self.mcts_search(state, mode)
        action, value = root_node.select_best_action(mode)

        return action, value

    @staticmethod
    def exponentiated_visit_counts(counts: List[int], total: int, temperature: float) -> np.array:
        exponentiated_visit_counts = np.array(counts) / total
        exponentiated_visit_counts = np.power(exponentiated_visit_counts, temperature)
        return exponentiated_visit_counts / sum(exponentiated_visit_counts)

    def stochastic_policy(self, state, temperature: float = 0.9, selection_mode='mean', exploration=False):

        root_node, stats = self.mcts_search(state)
        visit_counts = [0] * self.env.max_num_actions()
        for c in root_node.children:
            visit_counts[c.action] = c.visits

        policy = self.exponentiated_visit_counts(visit_counts, root_node.visits, temperature)
        value = root_node.returns / root_node.visits

        if exploration:
            action = choices([i for i in range(len(policy))], policy)
        else:
            action = root_node.select_best_action(mode=selection_mode)[0]
        return policy, value, action, stats

    def __str__(self):
        return "MCTS(" + str(self.tree_policy) + ", " + str(self.expansion_policy) + ", " \
               + str(self.evaluation_policy) + ") " + str(self.evaluate_leaf_children) + " " \
               + str(self.initialize_tree) + " " + str(self.value_initialization)


if __name__ == '__main__':
    from envs.tsp.TSP import TSPGym, TSP
    # env = TSPGym(num_cities=15)
    # model = TSP(num_cities=15)
    #
    #
    # from tree_policies.tree_policy import UCTPolicy
    # from tree_policies.exploration_terms.puct_term import PUCTTerm
    # from tree_policies.exploitation_terms.avg_node_value import AvgNodeValueTerm
    # from evaluation_policies.neural_value_eval import NeuralValueEvalPolicy
    # from stable_baselines3 import PPO
    # from model_free.stb3_wrapper import Stb3ACAgent
    #
    # model_free_agent = PPO.load("results/trained_agents/tsp/model_free/ppo_tsp_15_3e6")
    # tp = UCTPolicy(AvgNodeValueTerm(), PUCTTerm(exploration_constant=1))
    # ep = ExpansionPolicy()
    # rp = NeuralValueEvalPolicy()
    # agent = MCTSAgent(model, tp, ep, rp, neural_net=Stb3ACAgent(model_free_agent), num_simulations=1000, dirichlet_noise=True)
    #
    # num_iter = 100
    #
    # rewards = 0
    # for _ in range(num_iter):
    #     state = env.reset()
    #     done = False
    #
    #     while not done:
    #         action, _ = agent.select_action(env.raw_state())
    #         state, reward, done, _ = env.step(action)
    #
    #     rewards += reward
    #
    #     env.render()
    # print("avg rew: ", rewards / num_iter)
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





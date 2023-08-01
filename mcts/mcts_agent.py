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
                 initialize_tree=True, persist_trajectories=False, **kwargs):
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
        self.persist_trajectories = persist_trajectories # whether to include rollout trajectories as nodes in the tree

    def init_prior(self, action_probs, new_children, s):
        if torch.is_tensor(action_probs) or action_probs:
            for i, c in enumerate(new_children):
                c.prior_prob = action_probs[i]
        else:
            self.tree_policy.exploration_term.init_prior(new_children, state=copy.deepcopy(s), env=self.env,
                                                         neural_net=self.neural_net)

    def init_tree(self, n: Node, s):
        """
        create a path from the root of the tree to a terminal node by greedily choosing nodes with the learned policy
        this should make the results of a tree search at least as good as the learned policy without tree search would have been
        but only under the assumption that value estimates are perfect
        """
        done = False

        while not done:
            new_children = self.expansion_policy.expand(n, s, model=self.model, env=self.env, neural_net=self.neural_net)
            state_value, action_probs, _ = self.evaluation_policy.evaluate(n, s, model=self.model, env=self.env, neural_net=self.neural_net)
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

    def mcts_search(self, state, mode='mean', root_node=None):
        model = ModelStepCounter(self.model)
        neural_net = EvalCounterWrapper(self.neural_net)

        if not root_node:
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

    def selection_phase(self, n: Node, s, model):
        done = False
        terminal_reward = None
        while not n.is_leaf():
            n = self.tree_policy.select(n, add_dirichlet=(n.is_root() and self.dirichlet_noise))
            s, terminal_reward, done = model.step(s, n.action)
            terminal_reward = self.env.reward(terminal_reward)
        return n, s, terminal_reward, done

    def expansion_phase(self, n: Node, s, model, neural_net):
        new_children = self.expansion_policy.expand(n, s, model=model, env=self.env, neural_net=neural_net)
        return new_children

    def evaluation_phase(self, n: Node, s, new_children, model, neural_net):
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
                state_values, action_probs, trajectories = self.evaluation_policy.evaluate_multiple(new_children, copy.deepcopy(states), model=model, env=self.env,
                                                                                  neural_net=neural_net)

            # todo: handle trajectories to persist in the tree here

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
            # only n is evaluated here, so we need to initialise its children somehow
            # initialise values of expanded children of n, either through value network or as inf
            if self.value_initialization:
                children_state_values = neural_net.state_values(
                    [self.env.observation(model.step(copy.deepcopy(s), c.action)[0]) for c in new_children])
                children_state_values = [c[0] for c in children_state_values]
            else:
                children_state_values = [math.inf for _ in new_children]

            for i, c in enumerate(new_children):
                c.returns = children_state_values[i]
                c.visits = 1

            # evaluate n
            state_value, action_probs, trajectory = self.evaluation_policy.evaluate(n, copy.deepcopy(s), model=model, env=self.env,
                                                                        neural_net=neural_net)

            # optionally persist trajectory in tree
            if trajectory: # in mixed policies trajectories may only be returned for some individual policies
                n = self.process_trajectory(n, trajectory)

            # backpropagate the evaluated value of n up the tree
            self.backpropagation_phase(n, state_value)

            # initialise prior probability for selection policy
            self.init_prior(action_probs, new_children, s)

    def process_trajectory(self, n: Node, trajectory: List[Node, ]):
        """
        Merge the trajectory from a rollout into the search tree
        @param n: node at which the rollout starts (already part of the tree)
        @param trajectory: list of nodes comprising the trajectory
        @return: the last node of the trajectory, if desired
        """
        if self.persist_trajectories:
            if trajectory and len(trajectory) > 1:
                # trajectory starts at n and second element of trajectory is a child already initialised above
                # kill and replace the child:
                child_action = trajectory[1][0].action

                child_indices = [i for i, c in enumerate(n.children) if c.action == child_action]

                if len(child_indices) == 0:
                    return n
                
                child_index = child_indices[0]
                n.children[child_index] = trajectory[1][0]

                # draw the rest of the edges in the trajectory
                for i, (n_, _) in enumerate(trajectory[1:-1]):
                    n_.children.append(trajectory[i + 2][0])

            # init prior probs in trajectory nodes
            for tn, ts in trajectory:
                if tn.action is not None:
                    self.tree_policy.exploration_term.init_prior([tn], state=ts, env=self.env, neural_net=self.neural_net)

            n = trajectory[-1][0]  # perform subsequent backpropagation from last node in trajectory

        return n

    def backpropagation_phase(self, n: Node, value: float):
        """
        Backpropagates a value up the tree starting at n
        """
        while n.has_parent():
            n.update(value)
            n = n.parent

        n.update(value)

    def select_action(self, state, mode='mean', root=None):
        root_node, _ = self.mcts_search(state, mode, root_node=root)
        action, value = root_node.select_best_action(mode)

        return action, value, root_node

    @staticmethod
    def exponentiated_visit_counts(counts: List[int], total: int, temperature: float) -> np.array:
        exponentiated_visit_counts = np.array(counts) / total
        exponentiated_visit_counts = np.power(exponentiated_visit_counts, temperature)
        return exponentiated_visit_counts / sum(exponentiated_visit_counts)

    def stochastic_policy(self, state, temperature: float = 0.9, selection_mode='mean', exploration=False, root=None):
        root_node, stats = self.mcts_search(state, root_node=root)
        visit_counts = [0] * self.env.max_num_actions()
        for c in root_node.children:
            visit_counts[c.action] = c.visits

        policy = self.exponentiated_visit_counts(visit_counts, root_node.visits, temperature)
        value = root_node.returns / root_node.visits

        if exploration:
            action = choices([i for i in range(len(policy))], policy)
        else:
            action = root_node.select_best_action(mode=selection_mode)[0]

        children_states, children_values = self.children_training_targets(root_node, state)

        return policy, value, action, stats, children_states, children_values, root_node

    def children_training_targets(self, root_node, root_state):
        root_state = copy.deepcopy(root_state)
        root_obs = self.env.observation(root_state)
        children_states = [root_obs] * self.env.max_num_actions() # we want to have consistent tensor sizes, but might not always have the same number of children.
        children_values = [root_node.value()] * self.env.max_num_actions() # as a workaround, we simply fill the missing children with the root node values
        for i, c in enumerate(root_node.children):
            children_states[i] = self.env.observation(self.model.step(root_state, c.action)[0])
            children_values[i] = c.value()

        return children_states, children_values

    def __str__(self):
        return "MCTS(" + str(self.tree_policy) + ", " + str(self.expansion_policy) + ", " \
               + str(self.evaluation_policy) + ") " + str(self.evaluate_leaf_children) + " " \
               + str(self.initialize_tree) + " " + str(self.value_initialization)






from mcts.node import Node
from mcts.expansion_policies.expansion_policy import ExpansionPolicy

import numpy as np


class NeuralExpansionPolicy(ExpansionPolicy):
    def __init__(self, threshold):
        self.threshold = threshold

    def expand(self, node, state, model, neural_net, env, **kwargs):
        new_children = []

        legal_actions = model.legal_actions(state)
        # todo the following is already called in the main mcts code. remove redundancy!
        # but: evaluate_actions always return probs summing to 1. if we compute probs for all legal actions
        # and then discard some of them, the remaining set will not sum to 1 anymore. rescaling them might still be
        # faster than calling the neural net again
        _, action_probs = neural_net.evaluate_actions(env.observation(state), legal_actions)

        selected_actions = self.select_actions(action_probs, legal_actions)

        for a in selected_actions:
            child = Node(a, node)
            new_children.append(child)
            node.children.append(child)

        return new_children

    def select_actions(self, action_probs, actions):
        """ Selects the minimal number of actions whose probabilities sum to the given threshold
        """
        actions = np.array(actions)
        action_probs = np.array(action_probs)
        descending_probabilities = np.sort(action_probs)[::-1]
        cumulative_probabilities = np.cumsum(descending_probabilities)
        threshold_met = cumulative_probabilities >= self.threshold - 1e-6
        prob_sum_below_threshold = np.argmax(threshold_met) + 1

        descending_prob_indices = np.argsort(action_probs)[::-1]
        indices_to_expand = descending_prob_indices[:prob_sum_below_threshold]
        actions_to_expand = actions[indices_to_expand]

        return actions_to_expand



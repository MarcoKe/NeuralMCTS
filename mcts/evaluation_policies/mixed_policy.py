from mcts.evaluation_policies.evaluation_policy import EvaluationPolicy
from mcts.node import Node
from typing import List
from itertools import compress
import numpy as np


class MixedPolicy(EvaluationPolicy):
    def __init__(self, policies: List[EvaluationPolicy], thresholds: List[int]):
        """
        @param policies: Evaluation policies to be used at different search depths
        @param thresholds: From which depth each evaluation policy is supposed to be used. First value should always be 0
        Example: policies = [a, b], thresholds = [0, 10] will use policy a for depths < 10 and policy b for depths >= 10
        """
        self.thresholds = np.array(thresholds)
        self.policies = policies

    def evaluate(self, node: Node, state, neural_net=None, model=None, env=None):
        real_node_depth = env.current_num_steps() + node.depth

        # select policy for corresponding node depth
        policy = list(compress(self.policies, self.thresholds <= real_node_depth))[-1]

        return policy.evaluate(node, state, neural_net=neural_net, model=model, env=env)

    def __str__(self):
        return "MixedEval"

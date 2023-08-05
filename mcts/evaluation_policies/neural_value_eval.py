from typing import List

from .evaluation_policy import EvaluationPolicy
from ..node import Node


class NeuralValueEvalPolicy(EvaluationPolicy):
    def evaluate(self, node: Node, state, neural_net=None, model=None, env=None):
        legal_actions = model.legal_actions(state)
        state_value, action_probs = neural_net.evaluate_actions(env.observation(state), legal_actions)

        return state_value, action_probs, None

    def evaluate_multiple(self, nodes: List[Node], states, neural_net=None, model=None, env=None):
        state_values = neural_net.state_values([env.observation(s) for s in states])
        state_values = [v[0] for v in state_values]
        return state_values, None, None

    def __str__(self):
        return "NeuralValueEval"

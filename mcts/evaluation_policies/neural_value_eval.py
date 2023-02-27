from .evaluation_policy import EvaluationPolicy


class NeuralValueEvalPolicy(EvaluationPolicy):
    def evaluate(self, state, neural_net=None, model=None, env=None):
        legal_actions = model.legal_actions(state)
        state_value, action_probs = neural_net.evaluate_actions(env.observation(state), legal_actions)

        return state_value, action_probs

    def evaluate_multiple(self, states, neural_net=None, model=None, env=None):
        state_values = neural_net.state_values([env.observation(s) for s in states])

        return state_values, None

    def __str__(self):
        return "NVRoll"

from .evaluation_policy import EvaluationPolicy


class NeuralValueEvalPolicy(EvaluationPolicy):
    def evaluate(self, state, neural_net=None, model=None, env=None):
        legal_actions = model.legal_actions(state)
        state_value, action_probs = neural_net.evaluate_actions(env.observation(state), legal_actions)

        return state_value, action_probs

    def __str__(self):
        return "NVRoll"

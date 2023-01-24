from .evaluation_policy import EvaluationPolicy


class NeuralValueEvalPolicy(EvaluationPolicy):
    def __init__(self, model_free_agent, model):
        self.agent = model_free_agent
        self.model = model

    def evaluate(self, state):
        legal_actions = self.model.legal_actions(state)
        state_value, action_probs = self.agent.evaluate_state(self.model.create_obs(state), legal_actions)

        return state_value, action_probs

    def __str__(self):
        return "NVRoll"

from mcts.evaluation_policies.evaluation_policy import EvaluationPolicy
import torch


class NeuralRolloutPolicy(EvaluationPolicy):
    def __init__(self, model_free_agent, model):
        self.agent = model_free_agent
        self.model = model

    def evaluate(self, state):
        done = False

        priors = None
        first_iteration = True
        while not done:
            legal_actions = self.model.legal_actions(state)
            _, action_probs = self.agent.evaluate_actions(self.model.create_obs(state), legal_actions)
            if first_iteration:
                priors = action_probs
                first_iteration = False

            action = torch.argmax(action_probs)
            state, reward, done = self.model.step(state, action)

        return reward, priors

    def __str__(self):
        return "NRoll"

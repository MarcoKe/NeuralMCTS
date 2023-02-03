from mcts.evaluation_policies.evaluation_policy import EvaluationPolicy
import torch


class NeuralRolloutPolicy(EvaluationPolicy):
    def evaluate(self, state, neural_net=None, model=None, env=None):
        done = False

        priors = None
        first_iteration = True
        while not done:
            legal_actions = model.legal_actions(state)
            _, action_probs = neural_net.evaluate_actions(env.observation(state), legal_actions)
            if first_iteration:
                priors = action_probs
                first_iteration = False

            action = torch.argmax(action_probs)
            state, reward, done = model.step(state, action)

        return reward, priors

    def __str__(self):
        return "NRoll"

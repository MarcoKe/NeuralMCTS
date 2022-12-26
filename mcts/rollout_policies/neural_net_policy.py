from mcts.rollout_policies.rollout_policy import RolloutPolicy
import torch


class NeuralRolloutPolicy(RolloutPolicy):
    def __init__(self, model_free_agent, model):
        self.agent = model_free_agent
        self.model = model  # environment model

    def stb3_policy_probs(self, obs, actions):
        obs_tensor = self.agent.policy.obs_to_tensor(obs)[0]
        # Return likelihood of taking each action
        return torch.exp(self.agent.policy.evaluate_actions(obs_tensor, torch.tensor(actions))[1])

    def rollout(self, state):
        done = False

        while not done:
            legal_actions = self.model.legal_actions(state)

            action = torch.argmax(self.stb3_policy_probs(self.model.create_obs(state), legal_actions))
            state, reward, done = self.model.step(state, action)

        return reward

    def __str__(self):
        return "NRoll"



class NeuralValueRolloutPolicy(RolloutPolicy):
    def __init__(self, model_free_agent, model):
        self.agent = model_free_agent
        self.model = model

    def stb3_value(self, obs, actions):
        obs_tensor = self.agent.policy.obs_to_tensor(obs)[0]
        # Return estimated value of each action given the observation
        return self.agent.policy.evaluate_actions(obs_tensor, torch.tensor(actions))[0]

    def rollout(self, state):

        legal_actions = self.model.legal_actions(state)

        return self.stb3_value(self.model.create_obs(state), legal_actions)

    def __str__(self):
        return "NVRoll"

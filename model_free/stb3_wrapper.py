import torch


class RLAgent:
    def evaluate_actions(self, obs, actions):
        raise NotImplementedError

    def state_values(self, observations):
        raise NotImplementedError


class Stb3ACAgent(RLAgent):
    def __init__(self, agent):
        self.agent = agent

    def evaluate_actions(self, obs, actions=None):
        if not actions:
            actions = [1]

        with torch.no_grad():
            obs_tensor = self.agent.policy.obs_to_tensor(obs)[0]
            value, logprob, entropy = self.agent.policy.evaluate_actions(obs_tensor, torch.tensor(actions))

        return value, torch.nn.functional.softmax(logprob, dim=0)

    def state_values(self, observations):

        with torch.no_grad():
            obs_tensor = self.agent.policy.obs_to_tensor(observations)[0]
            values = self.agent.policy.predict_values(obs_tensor)

        return values.tolist()

    def select_action(self, obs, legal_actions):
        _, dist = self.evaluate_actions(obs, legal_actions)

        return legal_actions[torch.argmax(dist)]


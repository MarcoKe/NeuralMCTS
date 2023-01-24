import torch


class RLAgent:
    def evaluate_state(self, obs, actions):
        raise NotImplementedError


class Stb3ACAgent(RLAgent):
    def __init__(self, agent):
        self.agent = agent

    def evaluate_state(self, obs, actions=None):
        if not actions:
            actions = [1]

        obs_tensor = self.agent.policy.obs_to_tensor(obs)[0]
        value, logprob, entropy = self.agent.policy.evaluate_actions(obs_tensor, torch.tensor(actions))
        return value, torch.exp(logprob)


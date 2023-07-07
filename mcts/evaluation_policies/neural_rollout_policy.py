from typing import List

from mcts.evaluation_policies.evaluation_policy import EvaluationPolicy
import torch

from mcts.node import Node


class NeuralRolloutPolicy(EvaluationPolicy):
    def evaluate(self, node: Node, state, neural_net=None, model=None, env=None):
        done = False
        trajectory: List[Node] = [node] # we can optionally persist any rollout trajectories in the search tree

        priors = None
        first_iteration = True
        while not done:
            legal_actions = model.legal_actions(state)
            _, action_probs = neural_net.evaluate_actions(env.observation(state), legal_actions)
            if first_iteration:
                priors = action_probs
                first_iteration = False

            all_action_probs = torch.Tensor([0.0 for _ in range(env.max_num_actions())])
            all_action_probs[legal_actions] = action_probs
            action = torch.argmax(all_action_probs)
            trajectory.append(Node(action, trajectory[-1]))

            state, reward, done = model.step(state, action)
            reward = env.reward(reward)

        return reward, priors, trajectory

    def __str__(self):
        return "NRoll"

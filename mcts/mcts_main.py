import copy
import random
import tqdm
import time
from tree_policies.tree_policy import TreePolicy, RandomTreePolicy, UCTPolicy
from tree_policies.neural_puct_policy import NeuralPUCTPolicy
from mcts.rollout_policies.rollout_policy import RolloutPolicy, RandomRolloutPolicy
from rollout_policies.neural_net_policy import NeuralRolloutPolicy
from stable_baselines3 import PPO
from envs.TSP import TSPGym, TSP
from node import Node
from tree_visualization.app import get_tree_visualization


class MCTSAgent:
    def __init__(self, model, tree_policy: TreePolicy, rollout_policy: RolloutPolicy, num_simulations=10):
        self.model = model
        self.tree_policy = tree_policy
        self.rollout_policy = rollout_policy
        self.num_simulations = num_simulations

    def select(self, node: Node, state):
        return self.tree_policy.select(node, state)

    def rollout(self, state):
        return self.rollout_policy.rollout(state)

    def mcts_search(self, state, mode='mean'):  # todo: ability to pass root node. that way you can continue where you left off
        root_node = Node(None, None)

        simulation_count = 0
        while simulation_count < self.num_simulations:
            simulation_count += 1

            n, s = root_node, copy.deepcopy(state)

            done = False
            while not n.is_leaf():
                n = self.select(n, s)  # uses tree policy
                # s contains the unscheduled actions and the tour
                s, _, done = self.model.step(s, n.action)  # todo make a distinction between model and env

            if not done:
                # Expand n with the unscheduled actions from s
                actions = self.model.legal_actions(s)
                n.expand(actions)
                # Initialize action probabilities
                action_probs = self.tree_policy.get_action_probs(s, actions)
                for i, c in enumerate(n.children):
                    c.action_prob = action_probs[i]
                # Select an action according to the tree policy
                n = self.select(n, s)
                # Rollout
                rollout_reward = self.rollout(s)

            while n.has_parent():
                # Backpropagation
                n.update(rollout_reward)
                n = n.parent

        action, value = root_node.select_best_action(mode)
        return action, value

    def select_action(self, state, mode='mean'):
        return self.mcts_search(state, mode)

    def __str__(self):
        return "MCTS(" + str(self.tree_policy) + ", " + str(self.rollout_policy) + ")"


if __name__ == '__main__':
    # Initialize environment and model
    env = TSPGym(num_cities=15)
    model = TSP(num_cities=15)

    # rewards = 0
    # for i in tqdm.tqdm(range(1000)):
    #     state = env.initial_problem()
    #     agent = MCTSAgent(env, tp, rp, num_simulations=1)
    #     done = False
    #
    #     while not done:
    #         action = agent.select_action(state)
    #         state, reward, done = env.step(state, action)
    #         # print(reward)
    #
    #     rewards += reward
    # print("avg rew: ", rewards / 1000)
    #
    # rewards = 0
    # for i in tqdm.tqdm(range(1000)):
    #     state = env.initial_problem()
    #     agent = MCTSAgent(env, tp, rp, num_simulations=20)
    #     done = False
    #
    #     while not done:
    #         action = agent.select_action(state)
    #         state, reward, done = env.step(state, action)
    #         # print(reward)
    #
    #     rewards += reward
    # print("avg rew: ", rewards / 1000)

    # Load trained agent
    agent = PPO.load("ppo_tsp_15")
    # rp = RandomRolloutPolicy(model)
    rp = NeuralRolloutPolicy(model_free_agent=agent, model=model)  # rollout policy
    # for i in tqdm.tqdm(range(1000)):
    state = env.reset()
    # tp = UCTPolicy(10)
    tp = NeuralPUCTPolicy(10, agent, model)  # tree policy
    agent = MCTSAgent(model, tp, rp, num_simulations=10000)
    done = False

    rewards = 0
    n = 1000  # number of iterations
    i = 0
    prev = time.time()
    while i < n:
        # print("select action")
        # prev = time.time()
        action, _ = agent.select_action(env.raw_state())
        # print(time.time() - prev)
        obs, reward, done, state = env.step(action)
        # print(state['tour'])
        if done:
            print("iteration " + str(i))
            print("time: " + str(time.time() - prev))
            prev = time.time()
            print("reward: " + str(reward))
            rewards += reward
            env.reset()
            i += 1

    # env.render()
    print("avg rew: ", rewards / n)

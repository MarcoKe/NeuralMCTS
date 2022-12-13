import copy
import random
import tqdm
import time
from tree_policies.tree_policy import TreePolicy, RandomTreePolicy
from tree_policies.neural_puct_policy import NeuralPUCTPolicy
from mcts.rollout_policies.rollout_policy import RolloutPolicy, RandomRolloutPolicy
from node import Node


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

    def mcts_search(self, state,
                    mode='mean'):  # todo: ability to pass root node. that way you can continue where you left off
        root_node = Node(None, None)

        simulation_count = 0
        while simulation_count < self.num_simulations:
            # print("Simulation " + str(simulation_count))
            simulation_count += 1

            n, s = root_node, copy.deepcopy(state)

            done = False
            while not n.is_leaf():
                n = self.select(n, s)  # uses tree policy
                # s contains the unscheduled actions and the tour
                s, _, done = self.model.step(s, n.action)  # todo make a distinction between model and env

            if not done:
                # Extend n with the unscheduled actions from s
                n.expand(self.model.legal_actions(s))
                # Select an action according to the tree policy
                n = self.select(n, s)  # uses tree policy
                rollout_reward = self.rollout(s)  # uses rollout policy

            while n.has_parent():
                n.update(rollout_reward)
                n = n.parent

        action, value = root_node.select_best_action(mode)
        return action, value

    def select_action(self, state, mode='mean'):
        return self.mcts_search(state, mode)

    def __str__(self):
        return "MCTS(" + str(self.tree_policy) + ", " + str(self.rollout_policy) + ")"


if __name__ == '__main__':
    from envs.TSP import TSPGym, TSP

    # tp = RandomTreePolicy()
    # Initialize environment and model
    env = TSPGym(num_cities=15)
    model = TSP(num_cities=15)
    # rp = RandomRolloutPolicy(model)

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

    rewards = 0
    from tree_policies.tree_policy import UCTPolicy
    from tree_policies.neural_puct_policy import NeuralPUCTPolicy

    # tp = UCTPolicy(10)
    from rollout_policies.neural_net_policy import NeuralRolloutPolicy
    from stable_baselines3 import PPO

    # Load trained agent
    agent = PPO.load("ppo_tsp_15")
    # Initialize rollout policy
    rp = NeuralRolloutPolicy(model_free_agent=agent, model=model)
    # for i in tqdm.tqdm(range(1000)):
    state = env.reset()
    # Initialize tree policy
    tp = NeuralPUCTPolicy(10, agent, model)
    agent = MCTSAgent(model, tp, rp, num_simulations=10000)
    done = False

    n = 10  # number of iterations
    # while not done:
    for i in range(n):
        start = time.time()
        action = agent.select_action(env.raw_state())  # runs mcts_search -> num_simulations times
        print(f'Action Selection Time: {time.time() - start}')
        state, reward, done, _ = env.step(action)
        # print(reward)

    rewards += reward

    # env.render()
    print("avg rew: ", rewards / 1000)

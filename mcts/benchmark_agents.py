import tqdm
import copy
import pandas as pd

from tree_policies.tree_policy import UCTPolicy
from rollout_policies.neural_net_policy import NeuralRolloutPolicy, NeuralValueRolloutPolicy
from rollout_policies.rollout_policy import RandomRolloutPolicy
from tree_policies.neural_puct_policy import NeuralPUCTPolicy
from stable_baselines3 import PPO
from mcts.mcts_main import MCTSAgent
import matplotlib.pyplot as plt

class Agent:
    def select_action(self, obs):
        raise NotImplementedError


class Stb3AgentWrapper(Agent):
    def __init__(self, agent):
        self.agent = agent

    def select_action(self, obs):
        action, _ = self.agent.predict(obs, deterministic=True)
        return action

    def __str__(self):
        return "Stb3"


class MCTSAgentWrapper(Agent):
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def select_action(self, obs):
        action, value = self.agent.select_action(self.env.raw_state())
        return action

    def __str__(self):
        return str(self.agent)


def evaluate_agent(env, agent, trials=10):
    rewards = 0
    for _ in tqdm.tqdm(range(trials)):
        reward = perform_episode(env, agent)
        rewards += reward

    env.render()
    print("avg rew: ", rewards / trials)


def perform_episode(env, agent, hstate=None, obs=None):
    if not hstate:
        state = env.reset()
    else:
        env.state = hstate
        state = obs

    done = False

    while not done:
        action = agent.select_action(state)  # todo: why throw the tree away after every action
        state, reward, done, _ = env.step(action)

    return reward


def compete(env, agents, trials=10):
    for i in tqdm.tqdm(range(trials)):
        state = copy.deepcopy(env.reset())
        state_ = copy.deepcopy(env.raw_state())
        rewards = []
        for agent in agents:
            reward = perform_episode(env, agent, state_, state)
            rewards.append(reward)
            env.render()

        print("rewards: ", [(str(a), r) for a, r in zip(agents, rewards)])


def budget_analysis(env, agents, trials=15):
    state = env.reset()
    state_ = copy.deepcopy(env.raw_state())

    trials = [10*t for t in range(trials)]
    df = pd.DataFrame(columns=['budget'] + [str(a) for a in agents])
    df['budget'] = trials

    for agent in tqdm.tqdm(agents):
        print(f"evaluating {agent}")
        rewards = []
        for c, i in enumerate(trials):
            if isinstance(agent, MCTSAgentWrapper):
                agent.agent.num_simulations = i
            else:
                if c > 0:
                    rewards = rewards * len(trials)
                    break

            reward = perform_episode(env, agent, hstate=state_, obs=state)
            rewards.append(reward)
            print("budget: ", i, " , reward: ", reward)
            env.render()

        df[str(agent)] = rewards

    return df

def plot_budget_analysis(df):
    agents = df.columns[1:]

    plt.figure()
    for a in agents:
        plt.plot(df['budget'], df[a], '--o', label=a)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    from envs.TSP import TSPGym, TSP

    env = TSPGym(num_cities=15)
    model = TSP(num_cities=15)
    model_free_agent = PPO.load("ppo_tsp")
    simulation_budget = 30

    # valueRolloutAgent = MCTSAgent(model, UCTPolicy(exploration_const=10),
    #                               NeuralValueRolloutPolicy(model_free_agent, model),
    #                               num_simulations=simulation_budget)
    #
    # evaluate_agent(env, valueRolloutAgent)

    nPUCTAgent = MCTSAgentWrapper(MCTSAgent(model, NeuralPUCTPolicy(exploration_const=1, agent=model_free_agent, model=model),
                                  RandomRolloutPolicy(model),
                                  num_simulations=simulation_budget), env)

    nPUCTValueRolloutAgent = MCTSAgentWrapper(MCTSAgent(model, NeuralPUCTPolicy(exploration_const=1, agent=model_free_agent, model=model),
                           NeuralValueRolloutPolicy(model_free_agent, model),
                           num_simulations=simulation_budget), env)

    nPUCTRolloutAgent = MCTSAgentWrapper(MCTSAgent(model,
                                       NeuralPUCTPolicy(exploration_const=1, agent=model_free_agent, model=model),
                                       NeuralRolloutPolicy(model_free_agent, model),
                                       num_simulations=simulation_budget), env)

    # evaluate_agent(env, nPUCTAgent)

    randomRolloutAgent = MCTSAgentWrapper(MCTSAgent(model, UCTPolicy(exploration_const=1),
                                   RandomRolloutPolicy(model),
                                   num_simulations=simulation_budget), env)


    # budget_analysis()
    ppo_agent = Stb3AgentWrapper(model_free_agent)
    # evaluate_agent(env, ppo_agent)
    # evaluate_agent(env, nPUCTAgent)
    # compete(env, [ppo_agent, nPUCTAgent, nPUCTRolloutAgent, nPUCTValueRolloutAgent])
    df = budget_analysis(env, [ppo_agent, nPUCTAgent, nPUCTRolloutAgent, nPUCTValueRolloutAgent], trials=100)
    df.to_csv("budget_analysis.csv")
    df = pd.read_csv("budget_analysis.csv", index_col=0)
    plot_budget_analysis(df)

    # neuralRolloutAgent = MCTSAgent(model, UCTPolicy(exploration_const=10),
    #                                NeuralRolloutPolicy(model_free_agent=model_free_agent, model=model),
    #                                num_simulations=simulation_budget)
    #
    # evaluate_agent(env, neuralRolloutAgent)


    # 2) todo dirichlet noise
    # 3) todo flat monte carl
    # 5) todo improve the model free starting point: cnn, gnn
    # 4) todo ranked reward
    # 1) todo proper experiment with visualization. all different configurations with different runtimes.
    # then scatter matrix with plotted tour and curves (x: budget, y, quality), multiple in one plot
    # 6) todo: mcts while training
    # 0) todo: refactor
    # todo optimize neural net usage efficiency
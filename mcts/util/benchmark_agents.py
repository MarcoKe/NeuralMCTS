import tqdm
import copy
import pandas as pd
import numpy as np
import seaborn as sns

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
        return "PPO"


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


def opt_gap(opt, sol):
    return (sol - opt) / opt


def budget_analysis(env, agents, trials=15, render=True):
    state = env.reset()
    state_ = copy.deepcopy(env.raw_state())

    trials = [10*t for t in range(trials)]
    df = pd.DataFrame(columns=['budget'] + [str(a) for a in agents])
    df['budget'] = trials

    from envs.tsp_solver import solve as solve_exactly
    optimum = solve_exactly(state_['unscheduled'])
    print('optimum: ', optimum)

    # df['optimum'] = [optimum] * len(trials)

    for agent in tqdm.tqdm(agents):
        print(f"evaluating {agent}")
        rewards = []
        for c, i in enumerate(trials):
            if isinstance(agent, MCTSAgentWrapper):
                agent.agent.num_simulations = i
            # else:
            #     if c > 0:
            #         rewards = rewards * len(trials)
            #         break

            reward = perform_episode(env, agent, hstate=state_, obs=state)
            rewards.append(opt_gap(optimum, abs(reward)))
            print("budget: ", i, " , reward: ", reward, " optimum: ", optimum, " gap: ", opt_gap(optimum, abs(reward)))
            if render: env.render()

        df[str(agent)] = rewards

    return df


def averaged_budget_analysis(env, agents, trials=10, num_repetitions=10):
    dfs = []
    for i in range(num_repetitions):
        df = budget_analysis(env, agents, trials, render=False)
        dfs.append(df)
        df.to_csv('df' + str(i) + '.csv')

    fig, ax = plt.subplots()
    clrs = sns.color_palette("Set2", 5)
    styles = ['o', '^', '<', 's', 'D', 'h']
    with sns.axes_style("darkgrid"):
        for i, agent in enumerate(agents):
            agent = str(agent)
            means =  np.array([np.mean(k) for k in zip(*[df[agent] for df in dfs])])
            errors = np.array([np.std(k, ddof=1) / np.sqrt(np.size(k)) for k in zip(*[df[agent] for df in dfs])])
            ax.plot(dfs[0]['budget'], means, '-o', label=agent, c=clrs[i], marker=styles[i])
            ax.fill_between(dfs[0]['budget'], means-errors, means+errors ,alpha=0.3, facecolor=clrs[i])
    plt.ylim([0.0, 1.4])
    plt.title('TSP Distance Minimization')
    plt.xlabel('Simulation Budget')
    plt.ylabel('Optimality Gap')
    plt.legend()
    plt.show()
    plt.savefig('budget_comparison.png')


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

    from mcts.tree_policies.tree_policy import UCTPolicy
    from mcts.tree_policies.exploration_terms.puct_term import PUCTTerm
    from mcts.tree_policies.exploitation_terms.avg_node_value import AvgNodeValueTerm
    from mcts.tree_policies.exploitation_terms.max_node_value import MaxNodeValueTerm

    from mcts.evaluation_policies.neural_value_eval import NeuralValueEvalPolicy
    from mcts.expansion_policies.expansion_policy import ExpansionPolicy
    from stable_baselines3 import PPO
    from model_free.stb3_wrapper import Stb3ACAgent

    model_free_agent = PPO.load("ppo_tsp_15_1e6")
    tp = UCTPolicy(AvgNodeValueTerm(), PUCTTerm(exploration_constant=1))
    ep = ExpansionPolicy(model=model)
    rp = NeuralValueEvalPolicy(model_free_agent=Stb3ACAgent(model_free_agent), model=model)
    agent = MCTSAgentWrapper(MCTSAgent(model, tp, ep, rp, num_simulations=1000), env)

    tp2 = UCTPolicy(AvgNodeValueTerm(), PUCTTerm(exploration_constant=1))
    ep2 = ExpansionPolicy(model=model)
    rp2 = NeuralValueEvalPolicy(model_free_agent=Stb3ACAgent(model_free_agent), model=model)
    agent2 = MCTSAgentWrapper(MCTSAgent(model, tp2, ep2, rp2, num_simulations=1000, dirichlet_noise=True), env)
    # tp2 = UCTPolicy(MaxNodeValueTerm(), PUCTTerm(exploration_constant=1))
    # ep2 = ExpansionPolicy(model=model)
    # rp2 = NeuralValueEvalPolicy(model_free_agent=Stb3ACAgent(model_free_agent), model=model)
    # agent2 = MCTSAgentWrapper(MCTSAgent(model, tp2, ep2, rp2, num_simulations=1000), env)


    # budget_analysis()
    ppo_agent = Stb3AgentWrapper(model_free_agent)
    # evaluate_agent(env, ppo_agent)
    # evaluate_agent(env, nPUCTAgent)
    # compete(env, [ppo_agent, nPUCTAgent, nPUCTRolloutAgent, nPUCTValueRolloutAgent])
    agents = [ppo_agent, agent, agent2]
    # df = budget_analysis(env, agents, trials=10)
    # df.to_csv("budget_analysis.csv")
    # df = pd.read_csv("budget_analysis.csv", index_col=0)
    # plot_budget_analysis(df)
    print(averaged_budget_analysis(env, agents))
    # neuralRolloutAgent = MCTSAgent(model, UCTPolicy(exploration_const=10),
    #                                NeuralRolloutPolicy(model_free_agent=model_free_agent, model=model),
    #                                num_simulations=simulation_budget)
    #
    # evaluate_agent(env, neuralRolloutAgent)


    # 3) todo flat monte carl
    # 5) todo improve the model free starting point: cnn, gnn
    # 6) todo: mcts while training
    # todo optimize neural net usage efficiency (turn off torch gradients?
    # todo: sensitivity analysis: how important is agent prediction quality. try saved agents from different stages of training
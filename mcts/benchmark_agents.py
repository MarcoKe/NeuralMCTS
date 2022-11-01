import tqdm
from tree_policies.tree_policy import UCTPolicy
from rollout_policies.neural_net_policy import NeuralRolloutPolicy, NeuralValueRolloutPolicy
from rollout_policies.rollout_policy import RandomRolloutPolicy
from tree_policies.neural_puct_policy import NeuralPUCTPolicy
from stable_baselines3 import PPO
from mcts.mcts_main import MCTSAgent

class Stb3Wrapper:
    def __init__(self, agent):
        self.agent = agent

    def select_action(self, obs):
        pass

def evaluate_agent(env, agent, trials=100):
    rewards = 0
    for i in tqdm.tqdm(range(trials)):
        state = env.reset()
        done = False

        while not done:
            action, value = agent.select_action(env.raw_state()) #todo: why throw the tree away after every action
            state, reward, done, _ = env.step(action)

        rewards += reward

    env.render()
    print("avg rew: ", rewards / trials)

def evaluate_stb3agent(env, agent, trials=100):
    rewards = 0
    for i in tqdm.tqdm(range(trials)):
        state = env.reset()
        done = False

        while not done:
            action, _ = agent.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)

        rewards += reward

    env.render()
    print("avg rew: ", rewards / trials)

def compete(env, agent, mf_agent, trials=10):
    for i in tqdm.tqdm(range(trials)):
        state = env.reset()
        import copy
        state_ =  copy.deepcopy(env.raw_state())
        done = False

        while not done:
            action, value = agent.select_action(env.raw_state()) #todo: why throw the tree away after every action
            state, reward, done, _ = env.step(action)
        mcts_reward = reward
        env.render()

        env.state = state_
        done = False

        while not done:
            action, _ = mf_agent.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)
        print("mcts: ", mcts_reward, " model free: ", reward)

if __name__ == '__main__':
    from envs.TSP import TSPGym, TSP

    env = TSPGym(num_cities=15)
    model = TSP(num_cities=15)
    model_free_agent = PPO.load("ppo_tsp")
    simulation_budget = 100

    # valueRolloutAgent = MCTSAgent(model, UCTPolicy(exploration_const=10),
    #                               NeuralValueRolloutPolicy(model_free_agent, model),
    #                               num_simulations=simulation_budget)
    #
    # evaluate_agent(env, valueRolloutAgent)

    nPUCTAgent = MCTSAgent(model, NeuralPUCTPolicy(exploration_const=1, agent=model_free_agent, model=model),
                                  RandomRolloutPolicy(model),
                                  num_simulations=simulation_budget)

    # evaluate_agent(env, nPUCTAgent)

    randomRolloutAgent = MCTSAgent(model, UCTPolicy(exploration_const=1),
                                   RandomRolloutPolicy(model),
                                   num_simulations=simulation_budget)

    # evaluate_agent(env, randomRolloutAgent)

    # evaluate_stb3agent(env, model_free_agent)

    compete(env, nPUCTAgent, model_free_agent)


    # neuralRolloutAgent = MCTSAgent(model, UCTPolicy(exploration_const=10),
    #                                NeuralRolloutPolicy(model_free_agent=model_free_agent, model=model),
    #                                num_simulations=simulation_budget)
    #
    # evaluate_agent(env, neuralRolloutAgent)
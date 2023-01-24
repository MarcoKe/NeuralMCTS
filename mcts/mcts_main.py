import copy
from mcts.tree_policies.tree_policy import TreePolicy
from mcts.evaluation_policies.evaluation_policy import EvaluationPolicy
from mcts.expansion_policies.expansion_policy import ExpansionPolicy
from mcts.node import Node


class MCTSAgent:
    def __init__(self, model, tree_policy: TreePolicy, expansion_policy: ExpansionPolicy, evaluation_policy: EvaluationPolicy,
                 num_simulations=10):
        self.model = model
        self.tree_policy = tree_policy
        self.expansion_policy = expansion_policy
        self.evaluation_policy = evaluation_policy
        self.num_simulations = num_simulations

    def mcts_search(self, state, mode='mean'): # todo: ability to pass root node. that way you can continue where you left off
        root_node = Node(None, None)

        simulation_count = 0
        while simulation_count < self.num_simulations:
            simulation_count += 1

            n, s = root_node, copy.deepcopy(state)

            done = False
            while not n.is_leaf():
                n = self.tree_policy.select(n)
                s, terminal_reward, done = self.model.step(s, n.action)

            if not done:
                new_children = self.expansion_policy.expand(n, s)
                state_value, action_probs = self.evaluation_policy.evaluate(s)
                for i, c in enumerate(new_children):
                    c.prior_prob = action_probs[i]
                    # experimental
                    s_, _, _ = self.model.step(state, c.action)
                    r_, _ = self.evaluation_policy.agent.evaluate_state(self.model.create_obs(s_))
                    c.update(r_)

                    # end exp

                # n = self.select(n, s)
            else:
                state_value = terminal_reward

            while n.has_parent():
                n.update(state_value)
                n = n.parent

        action, value = root_node.select_best_action(mode)
        return action, value

    def select_action(self, state, mode='max'):
        return self.mcts_search(state, mode)

    def __str__(self):
        return "MCTS(" + str(self.tree_policy) + ", " + str(self.evaluation_policy) + ")"


if __name__ == '__main__':
    from envs.TSP import TSPGym, TSP
    env = TSPGym(num_cities=15)
    model = TSP(num_cities=15)


    from tree_policies.tree_policy import UCTPolicy
    from tree_policies.exploration_terms.puct_term import PUCTTerm
    from tree_policies.exploitation_terms.avg_node_value import AvgNodeValueTerm
    from evaluation_policies.neural_value_eval import NeuralValueEvalPolicy
    from stable_baselines3 import PPO
    from model_free.stb3_wrapper import Stb3ACAgent

    model_free_agent = PPO.load("ppo_tsp")
    tp = UCTPolicy(AvgNodeValueTerm(), PUCTTerm(exploration_constant=1))
    ep = ExpansionPolicy(model=model)
    rp = NeuralValueEvalPolicy(model_free_agent=Stb3ACAgent(model_free_agent), model=model)
    agent = MCTSAgent(model, tp, ep, rp, num_simulations=1000)

    num_iter = 100


    rewards = 0
    for _ in range(num_iter):
        state = env.reset()
        done = False

        while not done:
            action, _ = agent.select_action(env.raw_state())
            state, reward, done, _ = env.step(action)

        rewards += reward

        env.render()
    print("avg rew: ", rewards / num_iter)


    rewards = 0
    for _ in range(num_iter):
        state = env.reset()
        done = False

        while not done:
            action, _ = model_free_agent.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)

        rewards += reward

        env.render()
    print("avg rew: ", rewards / num_iter)






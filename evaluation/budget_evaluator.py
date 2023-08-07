from mcts.node import Node
from mcts.util.benchmark_agents import MCTSAgentWrapper, Stb3AgentWrapper
import copy
import multiprocess as mp


class MCTSBudgetEvaluator:
    def __init__(self, exp_name, eval_env, mcts_agent, model_free_agent, budgets, wandb_run):
        self.exp_name = exp_name
        self.eval_env = eval_env
        self.mcts_agent = mcts_agent
        self.model_free_agent = model_free_agent
        self.budgets = budgets
        self.wandb_run = wandb_run
        self.eval_instances = 100 # todo: pass this as an argument

    def log(self, key, value, budget, instance):
        self.wandb_run.log({key: value, 'mcts_budget': budget, 'instance': instance})

    def evaluate(self):
        for b in self.budgets:
            self.mcts_agent.num_simulations = b

            self.evaluate_parallel(b)

    def evaluate_parallel(self, budget, workers=8):
        """
        Performs an evaluation of the current agent on instances provided by the eval_env instance generator specified
        in the environment config file. If multiple workers, the evaluation is executed in parallel.
        @param eval_iterations: number of instances to be evaluated.
        """
        instances = [(self.eval_env.generator.generate(),) for _ in range(self.eval_instances)]
        pool = mp.Pool(workers)
        results = pool.starmap(self.evaluate_single, instances)
        pool.close()

        for r in results:
            self.log('eval/rew_mcts', r[0], budget, r[1])

    def evaluate_single(self, instance):
        """
        Evaluation on a single problem instance using multiple methods (problem-specific solver, model-free, mcts)
        @param instance: the instance to be evaluated on
        @return: optimality gaps of all methods, difference in rewards between model free and mcts, mcts reward,
                 model free reward, instance id
        """

        # model free and mcts
        eval_env_ = copy.deepcopy(self.eval_env)
        self.mcts_agent.env = eval_env_

        reward_mcts = self.perform_eval_episode(eval_env_, MCTSAgentWrapper(self.mcts_agent, eval_env_),
                                                copy.deepcopy(instance))
        return reward_mcts, eval_env_.instance.id

    def perform_eval_episode(self, env, agent, instance):
        """
        Performs one evaluation episode by setting a specific problem instance in the environment
        """

        if agent.agent.num_simulations == 0:
            agent = Stb3AgentWrapper(agent.agent.neural_net.agent, env, env.model)

        state = env.set_instance(instance)
        state = env.observation(state)
        done = False

        steps = 0
        node = None
        while not done:
            action, node = agent.select_action(state, node)
            state, reward, done, _ = env.step(action)
            node = Node.create_root(node, action)
            steps += 1

        return reward
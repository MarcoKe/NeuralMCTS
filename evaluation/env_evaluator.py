from mcts.node import Node
from mcts.util.benchmark_agents import MCTSAgentWrapper
import copy
import multiprocess as mp
from experiment_management.utils import create_env, create_agent, init_wandb


class EnvEvaluator:
    def __init__(self, general_config, exp_name, exp_config, agent_config, env_configs):
        self.exp_name = exp_name
        self.general_config = general_config
        self.exp_config = exp_config
        self.agent_config = agent_config
        self.env_configs = env_configs

    def log(self, key, value, instance):
        self.wandb_run.log({key: value, 'instance': instance, 'env': self.entropy})

    def evaluate(self):
        for env_config in self.env_configs:
            self.entropy = env_config['entropy']
            self.eval_instances = env_config['params']['instance_generator_eval']['params']['max_instances']
            self.wandb_run = init_wandb(self.general_config, self.exp_name + '_' + self.entropy, self.exp_config, self.agent_config, env_config)

            _, eval_env, model = create_env(env_config)
            self.mcts_agent, model_free_agent = create_agent(self.general_config, eval_env, model, self.agent_config)
            self.eval_env = eval_env
            self.evaluate_parallel()
            self.wandb_run.finish()

    def evaluate_parallel(self, workers=8):
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
            self.log('eval/rew_mcts', r[0], r[1])

    def evaluate_single(self, instance):
        """
        Evaluation on a single problem instance using multiple methods (problem-specific solver, model-free, mcts)
        @param instance: the instance to be evaluated on
        @return: optimality gaps of all methods, difference in rewards between model free and mcts, mcts reward,
                 model free reward, instance id
        """

        eval_env_ = copy.deepcopy(self.eval_env)
        self.mcts_agent.env = eval_env_

        reward_mcts = self.perform_eval_episode(eval_env_, MCTSAgentWrapper(self.mcts_agent, eval_env_),
                                                copy.deepcopy(instance))
        return reward_mcts, eval_env_.instance.id

    def perform_eval_episode(self, env, agent, instance):
        """
        Performs one evaluation episode by setting a specific problem instance in the environment
        """

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
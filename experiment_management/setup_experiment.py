import wandb
import torch
import numpy as np
from envs.env_factory import env_factory, model_factory, instance_factories, observation_factories, action_factories, \
    reward_factories, solver_factories
from experiment_management.config_handling.load_exp_config import load_exp_config, load_sensitivity_exp_config
from stable_baselines3 import PPO
#from sb3_contrib import MaskablePPO as PPO
from sb3_contrib.common.wrappers import ActionMasker
from mcts.mcts_agent import MCTSAgent
from model_free.stb3_wrapper import Stb3ACAgent
from model_free.gnn_feature_extractor import GNNExtractor
from mcts.tree_policies.tree_policy_factory import tree_policy_factory
from mcts.expansion_policies.expansion_policy_factory import expansion_policy_factory
from mcts.evaluation_policies.eval_policy_factory import eval_policy_factory
from training.pimcts_trainer import MCTSPolicyImprovementTrainer
from training.stb3_trainer import Stb3Trainer
from evaluation.budget_evaluator import MCTSBudgetEvaluator


def env_from_config(env_config):
    observation_spaces = observation_factories.get(env_config['name'])
    action_spaces = action_factories.get(env_config['name'])
    reward_functions = reward_factories.get(env_config['name'])

    environment = env_factory.get(env_config['name'], **env_config['params'])
    environment = action_spaces.get(env_config['params']['action_space']['name'], env=environment)
    environment = reward_functions.get(env_config['params']['reward_function']['name'], env=environment)
    environment = observation_spaces.get(env_config['params']['observation_space']['name'],
                                         env=environment)  # this one needs to be last, do not change

    return environment


def create_env(env_config):
    instance_generators = instance_factories.get(env_config['name'])

    if 'instance_generator' in env_config['params']:
        gen_config = env_config['params']['instance_generator']
        env_config['params']['instance_generator'] = instance_generators.get(gen_config['name'], **gen_config['params'])

    environment = env_from_config(env_config)

    eval_env_config = env_config
    if 'instance_generator_eval' in env_config['params']:
        gen_config = env_config['params']['instance_generator_eval']
        eval_env_config['params']['instance_generator'] = instance_generators.get(gen_config['name'], **gen_config['params'])

    eval_environment = env_from_config(eval_env_config)

    model = model_factory.get(env_config['name'], **env_config['params'])

    return environment, eval_environment, model


def make_compatible(agent_config):
    if not 'persist_trajectories' in agent_config:
        agent_config['persist_trajectories'] = False

    return agent_config


def create_model_free_agent(general_config, env, config):
    if len(config['location']) > 0:
        return PPO.load(config['location'], env=env, tensorboard_log=general_config['output']['tensorboard_logs'])
    else:
        policy_kwargs = dict()
        policy_kwargs['activation_fn'] = torch.nn.modules.Mish
        learning_rate = 0.0001 if not 'learning_rate' in config else config['learning_rate']
        if 'net_arch' in config:
            policy_kwargs['net_arch'] = [config['net_arch']]
        if 'features_extractor' in config and config['features_extractor'] == 'gnn':
            feature_extractor_kwargs = dict(num_layers=3, num_mlp_layers=2, input_dim=2,
                                            hidden_dim=64, graph_pool="avg")
            policy_kwargs['features_extractor_class'] = GNNExtractor
            policy_kwargs['features_extractor_kwargs'] = feature_extractor_kwargs

        return PPO('MlpPolicy', env, learning_rate=learning_rate, tensorboard_log=general_config['output']['tensorboard_logs'], policy_kwargs=policy_kwargs)


def create_agent(general_config, env, model, agent_config):
    model_free_agent = create_model_free_agent(general_config, env, agent_config['learned_policy'])
    neural_net = Stb3ACAgent(model_free_agent)

    agent_config = make_compatible(agent_config)
    tp = tree_policy_factory.get(agent_config['tree_policy']['name'], **agent_config['tree_policy']['params'])
    ep = expansion_policy_factory.get(agent_config['expansion_policy']['name'], **agent_config['expansion_policy']['params'])
    rp = eval_policy_factory.get(agent_config['eval_policy']['name'], **agent_config['eval_policy']['params'])
    mcts_agent = MCTSAgent(env, model, tp, ep, rp, neural_net=neural_net,
                           num_simulations=agent_config['num_simulations'],
                           evaluate_leaf_children=agent_config['evaluate_leaf_children'],
                           value_initialization=agent_config['value_initialization'],
                           initialize_tree=agent_config['initialize_tree'],
                           persist_trajectories=agent_config['persist_trajectories'])

    return mcts_agent, model_free_agent


def init_wandb(general_config, exp_name, exp_config, agent_config, env_config):
    config = {'exp_name': exp_name, 'exp_config': exp_config, 'agent_config': agent_config, 'env_config': env_config}
    tag = 'test' if not 'tag' in exp_config else exp_config['tag']
    wandb.require("service")

    run = wandb.init(
        project=general_config['wandb']['project'],
        config=config,
        name=exp_config['name'],
        tags=[tag]
    )

    return run


def time_stats(start, stop, steps):
    total_time = stop - start
    avg_time_step = total_time / int(steps)
    return total_time, avg_time_step


def setup_experiment(exp_name):
    general_config, exp_name, exp_config, agent_config, env_config = load_exp_config(exp_name)
    wandb_run = init_wandb(general_config, exp_name, exp_config, agent_config, env_config)

    env, eval_env, model = create_env(env_config)

    mcts_agent, model_free_agent = create_agent(general_config, env, model, agent_config)

    solver_factory = solver_factories.get(env_config['name'])
    solver = solver_factory.get('opt')  # todo
    trainer = MCTSPolicyImprovementTrainer(exp_config['name'], env, eval_env, mcts_agent, model_free_agent, wandb_run=wandb_run, solver=solver,
                                           **agent_config['training'], policy_improvement_iterations=exp_config['policy_improvement_iterations'])

    trainer.train()
    wandb_run.finish()


def setup_budget_sensitivity_experiment(exp_name):
    general_config, exp_name, exp_config, original_exp, agent_config, env_config = load_sensitivity_exp_config(exp_name)
    if 'saved_agent_path' in exp_config:
        agent_config['learned_policy']['location'] = exp_config['saved_agent_path']

    general_config['wandb']['project'] = 'neural_mcts_budget'
    wandb_run = init_wandb(general_config, exp_name, exp_config, agent_config, env_config)

    _, eval_env, model = create_env(env_config)
    mcts_agent, model_free_agent = create_agent(general_config, eval_env, model, agent_config)

    evaluator = MCTSBudgetEvaluator(exp_config['name'], eval_env, mcts_agent, model_free_agent, exp_config['budgets'], wandb_run)

    evaluator.evaluate()
    wandb_run.finish()


def setup_model_free_experiment(exp_name):
    general_config, exp_name, exp_config, agent_config, env_config = load_exp_config(exp_name)
    wandb.tensorboard.patch(root_logdir=general_config['output']['tensorboard_logs'])
    wandb_run = init_wandb(general_config, exp_name, exp_config, agent_config, env_config)

    env, eval_env, _ = create_env(env_config)
    env.set_run(wandb_run)
    eval_env.set_run(wandb_run)

    def mask_fn(env) -> np.ndarray:
        mask = np.array([False for _ in range(env.max_num_actions())])
        mask[env.model.legal_actions(env.raw_state())] = True
        return mask

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    eval_env = ActionMasker(eval_env, mask_fn)  # Wrap to enable masking

    agent = create_model_free_agent(general_config, env, agent_config)

    trainer = Stb3Trainer(exp_config['name'], env, eval_env, agent, wandb_run, exp_config['training_steps'])
    trainer.train()
    trainer.evaluate()

    wandb_run.finish()


if __name__ == '__main__':
    # setup_budget_sensitivity_experiment('budget_sensitivity/budget_sensitivity_test')
    setup_experiment("15x15/jsp_puct_neural_expansion_random")
    # setup_experiment("jsp_test")
    # setup_model_free_experiment("model_free/jsp_test")
    # exps = ['model_free/inter_instance_op_02', 'model_free/inter_instance_op_03', 'model_free/inter_instance_op_04',
    #         'model_free/inter_instance_op_05', 'model_free/inter_instance_op_06', 'model_free/inter_instance_op_07',
    #         'model_free/inter_instance_op_08']
    #
    # for exp in exps:
    #     print(exp)
    #     setup_model_free_experiment(exp)
    # setup_experiment("naive2_ff_05/jsp_uct_neural_expansion_neural_rollout_eval_value_initialization_initialize_tree")

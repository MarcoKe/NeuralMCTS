from envs.env_factory import env_factory, model_factory, instance_factories, observation_factories, action_factories, \
    reward_factories, solver_factories
from mcts.mcts_agent import MCTSAgent
from model_free.stb3_wrapper import Stb3ACAgent
from model_free.gnn_feature_extractor import GNNExtractor
from mcts.tree_policies.tree_policy_factory import tree_policy_factory
from mcts.expansion_policies.expansion_policy_factory import expansion_policy_factory
from mcts.evaluation_policies.eval_policy_factory import eval_policy_factory
#from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO as PPO
import wandb
import torch
import uuid


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


def create_model_free_agent(general_config, env, config, exp_name):
    if len(config['location']) > 0:
        return PPO.load(config['location'], env=env, tensorboard_log=general_config['output']['tensorboard_logs'])
    else:
        policy_kwargs = dict()
        policy_kwargs['activation_fn'] = torch.nn.modules.Mish
        learning_rate = 0.0001 if not 'learning_rate' in config else config['learning_rate']
        clip_range = 0.2 if not 'clip_range' in config else config['clip_range']
        if 'net_arch' in config:
            policy_kwargs['net_arch'] = config['net_arch']
        if 'features_extractor' in config and config['features_extractor'] == 'gnn':
            feature_extractor_kwargs = dict(num_layers=3, num_mlp_layers=2, input_dim=2,
                                            hidden_dim=64, graph_pool="avg")
            policy_kwargs['features_extractor_class'] = GNNExtractor
            policy_kwargs['features_extractor_kwargs'] = feature_extractor_kwargs

        return PPO('MlpPolicy', env, learning_rate=learning_rate, clip_range=clip_range, tensorboard_log=general_config['output']['tensorboard_logs'] + '/' +exp_name, policy_kwargs=policy_kwargs)


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
    config = {'exp_name': exp_name , 'exp_config': exp_config, 'agent_config': agent_config, 'env_config': env_config}
    tag = 'test' if not 'tag' in exp_config else exp_config['tag']
    wandb.require("service")

    wandb_args = {'project': general_config['wandb']['project'], 'config': config, 'name': config['exp_name'], 'tags': [tag]}
    if 'wandb_id' in exp_config: # in case we have a multi-part experiment, runs can be resumed
        print("Resuming wandb run ", exp_config['wandb_id'])
        wandb_args['id'] = exp_config['wandb_id']
        wandb_args['resume'] = 'must'

    run = wandb.init(**wandb_args)

    return run


def time_stats(start, stop, steps):
    total_time = stop - start
    avg_time_step = total_time / int(steps)
    return total_time, avg_time_step
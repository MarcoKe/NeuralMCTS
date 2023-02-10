import wandb
import torch
from envs.env_factory import env_factory, model_factory, instance_factories, observation_factories, action_factories, \
    reward_factories, solver_factories
from experiment_management.config_handling.load_exp_config import load_exp_config
from stable_baselines3 import PPO
from mcts.mcts_agent import MCTSAgent
from model_free.stb3_wrapper import Stb3ACAgent
from mcts.tree_policies.tree_policy_factory import tree_policy_factory
from mcts.expansion_policies.expansion_policy_factory import expansion_policy_factory
from mcts.evaluation_policies.eval_policy_factory import eval_policy_factory
from training.pimcts_trainer import MCTSPolicyImprovementTrainer


def create_env(env_config):
    observation_spaces = observation_factories.get(env_config['name'])
    action_spaces = action_factories.get(env_config['name'])
    reward_functions = reward_factories.get(env_config['name'])
    instance_generators = instance_factories.get(env_config['name'])

    if 'instance_generator' in env_config['params']:
        gen_config = env_config['params']['instance_generator']
        env_config['params']['instance_generator'] = instance_generators.get(gen_config['name'], **gen_config['params'])

    environment = env_factory.get(env_config['name'], **env_config['params'])
    environment = action_spaces.get(env_config['params']['action_space']['name'], env=environment)
    environment = reward_functions.get(env_config['params']['reward_function']['name'], env=environment)
    environment = observation_spaces.get(env_config['params']['observation_space']['name'], env=environment) # this one needs to be last
    model = model_factory.get(env_config['name'], **env_config['params'])
    return environment, model


def create_agent(env, model, agent_config):
    if len(agent_config['learned_policy']['location']) > 0:
        model_free_agent = PPO.load(agent_config['learned_policy']['location'], env=env)
    else:
        model_free_agent = PPO('MlpPolicy', env, policy_kwargs=dict(activation_fn=torch.nn.modules.Mish))
    neural_net = Stb3ACAgent(model_free_agent)

    tp = tree_policy_factory.get(agent_config['tree_policy']['name'], **agent_config['tree_policy']['params'])
    ep = expansion_policy_factory.get(agent_config['expansion_policy']['name'], **agent_config['expansion_policy']['params'])
    rp = eval_policy_factory.get(agent_config['eval_policy']['name'], **agent_config['eval_policy']['params'])
    mcts_agent = MCTSAgent(env, model, tp, ep, rp, neural_net=neural_net, num_simulations=agent_config['num_simulations'])

    return mcts_agent, model_free_agent


def init_wandb(general_config, exp_name, exp_config, agent_config, env_config):
    config = {'exp_name': exp_name, 'exp_config': exp_config, 'agent_config': agent_config, 'env_config': env_config}
    wandb.require("service")

    run = wandb.init(
        project=general_config['wandb']['project'],
        config=config,
        name=exp_config['name']
        # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    # wandb.tensorboard.patch(save=False)

    return run


def time_stats(start, stop, steps):
    total_time = stop - start
    avg_time_step = total_time / int(steps)
    return total_time, avg_time_step


def setup_experiment(exp_name):
    general_config, exp_name, exp_config, agent_config, env_config = load_exp_config(exp_name)
    wandb_run = init_wandb(general_config, exp_name, exp_config, agent_config, env_config)

    env, model = create_env(env_config)

    mcts_agent, model_free_agent = create_agent(env, model, agent_config)

    solver_factory = solver_factories.get(env_config['name'])
    solver = solver_factory.get('opt')  # todo
    trainer = MCTSPolicyImprovementTrainer(exp_config['name'], env, mcts_agent, model_free_agent, wandb_run=wandb_run, solver=solver,
                                           **agent_config['training'], policy_improvement_iterations=exp_config['policy_improvement_iterations'])

    trainer.train()

    wandb_run.finish()


if __name__ == '__main__':
    # setup_experiment("20230128_exp_001")
    setup_experiment("jsp_001")

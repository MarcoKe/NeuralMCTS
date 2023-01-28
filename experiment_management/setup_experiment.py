import wandb
from envs.env_factory import env_factory, model_factory
from experiment_management.config_handling.load_exp_config import load_exp_config
from stable_baselines3 import PPO
from mcts.mcts_agent import MCTSAgent
from model_free.stb3_wrapper import Stb3ACAgent
from mcts.tree_policies.tree_policy_factory import tree_policy_factory
from mcts.expansion_policies.expansion_policy_factory import expansion_policy_factory
from mcts.evaluation_policies.eval_policy_factory import eval_policy_factory
from training.pimcts_trainer import MCTSPolicyImprovementTrainer


def create_env(env_config):
    environment = env_factory.get(env_config['name'], **env_config['params'])
    model = model_factory.get(env_config['name'], **env_config['params'])
    return environment, model


def create_agent(env, model, agent_config):
    agent_config = agent_config['params']
    model_free_agent = PPO.load(agent_config['learned_policy']['location'], env=env)
    neural_net = Stb3ACAgent(model_free_agent)

    tp = tree_policy_factory.get(agent_config['tree_policy']['name'], **agent_config['tree_policy']['params'])
    ep = expansion_policy_factory.get(agent_config['expansion_policy']['name'], **agent_config['expansion_policy']['params'])
    rp = eval_policy_factory.get(agent_config['eval_policy']['name'], **agent_config['eval_policy']['params'])
    mcts_agent = MCTSAgent(model, tp, ep, rp, neural_net=neural_net, num_simulations=agent_config['num_simulations'])

    return mcts_agent, model_free_agent


def init_wandb(general_config, exp_name, exp_config, agent_config, env_config):
    config = {'exp_name': exp_name, 'exp_config': exp_config, 'agent_config': agent_config, 'env_config': env_config}
    wandb.require("service")

    run = wandb.init(
        project=general_config['wandb']['project'],
        config=config,
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
    trainer = MCTSPolicyImprovementTrainer(env, mcts_agent, model_free_agent, wandb_run=wandb_run)
    trainer.train()

    wandb_run.finish()

if __name__ == '__main__':
    setup_experiment("exp_001")
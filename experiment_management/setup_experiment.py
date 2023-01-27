from pathlib import Path
import time
import os
import wandb
#
# from src.modules.utils.factories.env_factory import environments
# from src.modules.utils.factories.simulation_factory import simulations
from experiment_management.config_handling.load_exp_config import load_exp_config
#
# def create_env(env_config, sim, wandb_run=None):
#     if wandb_run: env_config['params']['wandb_run'] = wandb_run
#     environment = environments.get(env_config['name'], sim=sim, **env_config['params'])
#
#     return environment
#
#
# def create_sim(exp_config):
#     if 'sim_params' in exp_config:
#         return simulations.get(exp_config['sim'], **exp_config['sim_params'])
#
#     return simulations.get(exp_config['sim'])
#
#
# def create_agent(general_config, exp_name, exp_config, env, eval_env, agent_config, environment, run, seed):
#     if agent_config['framework'] == 'stb3':
#         from src.modules.utils.factories.stb3.agent_factory import agents
#         from src.modules.utils.factories.stb3.policy_factory import policies
#         from src.modules.utils.factories.stb3.create_callbacks import create_callbacks
#
#     # policy = policies.get(agent_config['agent_type'], agent_config['policy'], env, **agent_config['model_params'])
#     # agent_config['model_params']['prioritized_replay_beta_iters'] = None
#
#     # tb_logdir = Path(general_config['output']['tensorboard_logs']) / Path(exp_name)
#     # tb_logdir = Path(general_config['output']['tensorboard_logs']) / Path(run.id)
#     tb_logdir = f"runs/{run.id}"
#
#
#
#     agent = agents.get(agent_config['agent_type'], agent_config['policy'], env, **agent_config['model_params'],
#                        tensorboard_log=tb_logdir, seed=seed)
#
#     if 'agent_path' in exp_config: # todo do properly
#         print("loading agent")
#         agent = agent.load(exp_config['agent_path'], env=env)
#
#     callbacks = create_callbacks(general_config, exp_name, exp_config, eval_env, run)
#     return agent, callbacks


def init_wandb(general_config, exp_name, exp_config, agent_config, env_config):
    config = {'exp_name': exp_name, 'exp_config': exp_config, 'agent_config': agent_config, 'env_config': env_config}
    wandb.require("service")

    run = wandb.init(
        project=general_config['wandb']['project'],
        # config=config,
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
    #
    # seed = 64923
    # wandb_run.log({"agent_seed": seed})

    # sim = create_sim(exp_config)
    # environment = create_env(env_config, sim, wandb_run=wandb_run)
    # eval_env = create_env(env_config, sim)
    #
    # agent, callbacks = create_agent(general_config, exp_name, exp_config, environment, eval_env, agent_config, environment, wandb_run, seed)

    # start_time = time.perf_counter()
    # agent.learn(total_timesteps=int(exp_config['training_steps']), callback=callbacks)
    # total_time, avg_time_step = time_stats(start_time, time.perf_counter(), exp_config['training_steps'])
    # print(f"Total time: {total_time/60} minutes. Average time per step: {avg_time_step} seconds.")
    #
    # # Save trained RL Agent
    # agent_path = Path(general_config['output']['saved_agents']) / Path(exp_name) # todo doubling wandb
    # os.makedirs(os.path.dirname(agent_path), exist_ok=True)
    # agent.save(agent_path)
    # todo save configs in results
    from envs.TSP import TSPGym, TSP
    from stable_baselines3 import PPO
    import torch
    from mcts.mcts_main import MCTSAgent
    from training.pimcts_trainer import MCTSPolicyImprovementTrainer
    num_cities = 15
    env = TSPGym(num_cities=num_cities)
    model = TSP(num_cities=15)

    # agent = PPO.load("ppo_tsp_15_1e6.zip")
    policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish)
    # model_free_agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log="stb3_tsp_tensorboard/", policy_kwargs=policy_kwargs)
    model_free_agent = PPO.load('results/trained_agents/tsp/model_free/ppo_tsp_15_1e6_ent.zip', env=env)
    from mcts.tree_policies.tree_policy import UCTPolicy
    from mcts.tree_policies.exploration_terms.puct_term import PUCTTerm
    from mcts.tree_policies.exploitation_terms.avg_node_value import AvgNodeValueTerm
    from mcts.expansion_policies.expansion_policy import ExpansionPolicy
    from mcts.evaluation_policies.neural_value_eval import NeuralValueEvalPolicy
    from mcts.evaluation_policies.neural_rollout_policy import NeuralRolloutPolicy
    from model_free.stb3_wrapper import Stb3ACAgent

    tp = UCTPolicy(AvgNodeValueTerm(), PUCTTerm(exploration_constant=1))
    ep = ExpansionPolicy(model=model)
    rp = NeuralRolloutPolicy(model_free_agent=Stb3ACAgent(model_free_agent), model=model)
    mcts_agent = MCTSAgent(model, tp, ep, rp, num_simulations=100)

    trainer = MCTSPolicyImprovementTrainer(env, mcts_agent, model_free_agent, wandb_run=wandb_run)
    trainer.train()


    # wandb_run.finish()

if __name__ == '__main__':
    setup_experiment("test")
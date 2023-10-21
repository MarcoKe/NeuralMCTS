import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import MlpExtractor
from pathlib import Path

from experiment_management.config_handling.load_exp_config import load_yml
from experiment_management.setup_experiment import create_env

import gym
import JSSEnv

# env_config = load_yml((Path('../data/config/envs/jsp_gnn_001.yml')))
env_config = load_yml((Path('../data/config/envs/jsp_minimal_002.yml')))
env, _, _ = create_env(env_config)
# env = gym.make('jss-v1', env_config={'instance_path': 'INSTANCE_PATH'})

policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish, net_arch=[dict(pi=[256, 256], vf=[256, 256])])
# model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00005,
#             tensorboard_log="results/tensorboard/stb3_jsp_lb_tensorboard/",
#             policy_kwargs=policy_kwargs)
model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.00005, #batch_size=512,
                    # tensorboard_log="results/tensorboard/stb3_jsp_tensorboard/",
                    policy_kwargs=policy_kwargs, seed=3)
model.set_random_seed(3)
model.learn(total_timesteps=1_000_000)
# model.save("results/trained_agents/minimal_jsp/model_free/ppo_jsp_gnn_10x10.zip")
# model = PPO.load("results/trained_agents/minimal_jsp/model_free/ppo_jsp_gnn_6x6.zip")
#
# print("reward, std =", evaluate_policy(model, env, n_eval_episodes=1000, deterministic=False))
#
# obs = env.reset()
# done = False
# while not done:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#
# env.render()

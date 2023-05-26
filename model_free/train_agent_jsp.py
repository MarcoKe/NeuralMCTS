import torch
from stable_baselines3 import PPO
from pathlib import Path
from envs.minimal_jsp_env.jsp_env import JobShopEnv
from gnn_feature_extractor import GNNExtractor

from experiment_management.config_handling.load_exp_config import load_yml
from experiment_management.setup_experiment import create_env

# env_config = load_yml((Path('../data/config/envs/jsp_minimal_001.yml')))
env_config = load_yml((Path('data/config/envs/jsp_gnn_001.yml')))
env, _, _ = create_env(env_config)

# feature_extractor_kwargs = dict(num_layers=3, num_mlp_layers=2, input_dim=2,
#                                 hidden_dim=64, graph_pool="avg")
# policy_kwargs = dict(activation_fn=torch.nn.modules.activation.ReLU,
#                      net_arch=[dict(pi=[256, 256], vf=[256, 256])],
#                      features_extractor_class=GNNExtractor,
#                      features_extractor_kwargs=feature_extractor_kwargs)
policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish, net_arch=[dict(pi=[256, 256], vf=[256, 256])])
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00005,
            tensorboard_log="results/tensorboard/stb3_gnn_jsp_tensorboard/",
            policy_kwargs=policy_kwargs)
# model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00005, tensorboard_log="results/tensorboard/stb3_jsp_tensorboard/", policy_kwargs=policy_kwargs)
model.learn(total_timesteps=3_000_000)
# model.save("results/trained_agents/tsp/model_free/ppo_tsp_15_3e6_ent.zip")
# model = PPO.load("ppo_tsp_15")
#
# obs = env.reset()
# done = False
# while not done:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#
# env.render()
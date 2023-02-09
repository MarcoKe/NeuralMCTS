import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from gnn_feature_extractor import GNNExtractor

from envs.TSP import TSPGym
from envs.JSSP import JSSPGym

env = JSSPGym(n_j=2, n_m=2)

feature_extractor_kwargs = dict(num_layers=3, num_mlp_layers=2,
                                hidden_dim=64, graph_pool="avg")
policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish,
                     features_extractor_class=GNNExtractor,
                     features_extractor_kwargs=feature_extractor_kwargs)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="stb3_jssp_gnn_tensorboard/",
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=5_000_000)
model.save("ppo_jssp_2x2_gnn.zip")

# policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish)
# model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="stb3_jssp_tensorboard/", policy_kwargs=policy_kwargs)
# model.learn(total_timesteps=5_000_000)
# model.save("ppo_jssp_2x2.zip")

# model = PPO.load("../mcts/ppo_jssp_2x2")
# par = model.get_parameters()
# for name, dict in par.items():
#     print(name, type(dict))
#     for key, value in dict.items():
#         print("   ", key, type(value))
#         print(value)

# obs = env.reset()
# done = False
# while not done:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#
# env.render()

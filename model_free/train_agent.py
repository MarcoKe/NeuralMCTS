import torch
from stable_baselines3 import PPO

from envs.minimal_jsp_env.jsp_model import JobShopModel
from envs.gnn_jsp_env.reward_functions.opt_gap import OptimalityGapReward
from envs.tsp.TSP import TSPGym
from envs.minimal_jsp_env.jsp_env import JobShopEnv
from envs.minimal_jsp_env.action_spaces.naive import NaiveActionSpace
from envs.minimal_jsp_env.observation_spaces.naive import NaiveObservationSpace
from envs.minimal_jsp_env.util.jsp_generation.single_instance_generator import SingleInstanceRandomGenerator
from envs.gnn_jsp_env.jsp_env import GNNJobShopEnv
from envs.gnn_jsp_env.action_spaces.gnn import GNNActionSpace
from envs.gnn_jsp_env.observation_spaces.gnn import GNNObservationSpace
from envs.gnn_jsp_env.util.jsp_generation.single_instance_generator import SingleInstanceRandomGenerator
from gnn_feature_extractor import GNNExtractor


generator = SingleInstanceRandomGenerator(num_jobs=6, num_operations=6, num_machines=6)
env = OptimalityGapReward(GNNActionSpace(GNNObservationSpace(GNNJobShopEnv(generator))))

feature_extractor_kwargs = dict(num_layers=3, num_mlp_layers=2, input_dim=2,
                                hidden_dim=64, graph_pool="avg")
policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish,
                     features_extractor_class=GNNExtractor,
                     features_extractor_kwargs=feature_extractor_kwargs)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="stb3_jssp_gnn_tensorboard/",
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=3_000_000)
model.save("ppo_jsp_2x2_gnn.zip")

# generator = SingleInstanceRandomGenerator(num_jobs=2, num_operations=2)
# env = OptimalityGapReward(NaiveActionSpace(NaiveObservationSpace(JobShopEnv(generator))))
#
# policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="results/tensorboard/stb3_jssp_tensorboard/",
#             policy_kwargs=policy_kwargs, ent_coef=0.05)
# model.learn(total_timesteps=3_000_000)
# model.save("ppo_jssp_2x2.zip")

# env = TSPGym(num_cities=15)
#
#
# policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="results/tensorboard/stb3_tsp_tensorboard/", policy_kwargs=policy_kwargs, ent_coef=0.05)
# model.learn(total_timesteps=3_000_000)
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
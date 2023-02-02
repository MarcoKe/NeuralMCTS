import torch
from stable_baselines3 import PPO
from pathlib import Path
from envs.minimal_jsp_env.jsp_env import JobShopEnv
from envs.minimal_jsp_env.action_spaces.naive_action_space import NaiveActionSpace
from envs.minimal_jsp_env.observation_spaces.naive_obs_space import NaiveObservationSpace
from envs.minimal_jsp_env.util.jsp_generation.multiple_instance_generator import MultipleInstanceGenerator

instance_path = 'C:/gitrepos/jsp_env/data/jsp_instances/6x6x6/'
jsp_generator = MultipleInstanceGenerator(instance_path, 'samsonov')
env = NaiveActionSpace(NaiveObservationSpace(JobShopEnv(jsp_generator)))
# env = TSPGym(num_cities=15)


policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="results/tensorboard/stb3_minimal_jsp_tensorboard/", policy_kwargs=policy_kwargs)
model.learn(total_timesteps=3_000_000)
model.save("results/trained_agents/minimal_jsp/model_free/ppo_jsp_6x6_3e6_multi.zip")
# model = PPO.load("ppo_tsp_15")
#
# obs = env.reset()
# done = False
# while not done:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#
# env.render()
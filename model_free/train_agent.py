import torch
from stable_baselines3 import PPO

from envs.TSP import TSPGym

env = TSPGym(num_cities=15)


policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="stb3_tsp_tensorboard/", policy_kwargs=policy_kwargs, ent_coef=0.05)
model.learn(total_timesteps=3_000_000)
model.save("ppo_tsp_15_3e6_ent.zip")
# model = PPO.load("ppo_tsp_15")
#
# obs = env.reset()
# done = False
# while not done:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#
# env.render()
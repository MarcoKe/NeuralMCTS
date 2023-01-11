import torch
from stable_baselines3 import PPO

from envs.TSP import TSPGym
from envs.JSSP import JSSPGym

env = JSSPGym(n_j=4, n_m=3)


policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="stb3_tsp_tensorboard/", policy_kwargs=policy_kwargs)
model.learn(total_timesteps=5_000_000)
model.save("ppo_jssp.zip")
model = PPO.load("ppo_jssp")

# obs = env.reset()
# done = False
# while not done:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#
# env.render()
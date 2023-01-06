import torch
from stable_baselines3 import PPO

from envs.TSP import TSPGymSA

env = TSPGymSA(num_cities=15)


policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="../stb3_tsp_tensorboard/", policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1_000_000)
model.save("ppo_tsp_sa.zip")

obs = env.reset()
done = False
while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done = env.step(action)

env.render()

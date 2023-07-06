import torch
from sb3_contrib import MaskablePPO as PPO
from pathlib import Path

from experiment_management.config_handling.load_exp_config import load_yml
from experiment_management.setup_experiment import create_env

env_config = load_yml((Path('data/config/envs/jsp_minimal_003.yml')))
env, _, _ = create_env(env_config)

policy_kwargs = dict(activation_fn=torch.nn.modules.activation.Mish, net_arch=[dict(pi=[256, 256], vf=[256, 256])])
import numpy as np

def mask_fn(env) -> np.ndarray:
    mask = np.array([False for _ in range(env.max_num_actions())])
    mask[env.model.legal_actions(env.raw_state())] = True
    return mask

from sb3_contrib.common.wrappers import ActionMasker

env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00005,
            tensorboard_log="results/tensorboard/stb3_gnn_jsp_tensorboard/",
            policy_kwargs=policy_kwargs)



model.learn(total_timesteps=3_000_000)
# print(model.policy)
model.save("results/trained_agents/jsp/ppo_6x6.zip")
# model = PPO.load("ppo_tsp_15")
#
# obs = env.reset()
# done = False
# while not done:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#
# env.render()
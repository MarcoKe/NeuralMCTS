from wandb.integration.sb3 import WandbCallback
from sb3_contrib.common.maskable.utils import get_action_masks


class Stb3Trainer:
    def __init__(self, exp_name, env, eval_env, agent, wandb_run, training_steps, eval_instances=50):
        self.exp_name = exp_name
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.wandb_run = wandb_run
        self.training_steps = training_steps
        self.eval_instances = eval_instances

        self.wandb_callback = WandbCallback(
            gradient_save_freq=100000,
            model_save_path=f"results/models/{self.wandb_run.id}",
            verbose=0,
        )

    def train(self):
        self.agent.learn(total_timesteps=self.training_steps, callback=[self.wandb_callback])

    def evaluate(self):
        mean_reward = 0
        for _ in range(self.eval_instances):
            obs = self.eval_env.reset()
            done = False
            while not done:
                action_masks = get_action_masks(self.eval_env)
                action, _state = self.agent.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)

            mean_reward += reward
            self.wandb_run.log({'eval/instance': self.eval_env.current_instance().id, 'eval/rew': reward})

        mean_reward /= self.eval_instances
        self.wandb_run.log({'eval/mean_reward': mean_reward})

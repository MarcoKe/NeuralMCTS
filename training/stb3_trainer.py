from wandb.integration.sb3 import WandbCallback


class Stb3Trainer:
    def __init__(self, exp_name, env, eval_env, agent, wandb_run, training_steps):
        self.exp_name = exp_name
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.wandb_run = wandb_run
        self.training_steps = training_steps

        self.wandb_callback = WandbCallback(
            gradient_save_freq=100000,
            model_save_path=f"results/models/{self.wandb_run.id}",
            verbose=0,
        )

    def train(self):
        self.agent.learn(total_timesteps=self.training_steps, callback=[self.wandb_callback])





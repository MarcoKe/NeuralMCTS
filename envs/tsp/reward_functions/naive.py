import gym


class NaiveReward(gym.RewardWrapper):
    def __init__(self, env):
        self.env = env
        super(NaiveReward, self).__init__(env)

    def reward(self, reward):
        return reward


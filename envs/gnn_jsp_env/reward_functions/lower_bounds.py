import gym


class LowerBoundsDifferenceReward(gym.RewardWrapper):
    def __init__(self, env):
        self.env = env
        super(LowerBoundsDifferenceReward, self).__init__(env)

    def reward(self, reward):
        return reward[1]

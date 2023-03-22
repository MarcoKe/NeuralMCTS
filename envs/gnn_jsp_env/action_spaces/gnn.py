import gym


class GNNActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super(GNNActionSpace, self).__init__(env)

        self.action_space = gym.spaces.Discrete(self.env.num_jobs)

    def action(self, action):
        return action

import gym


class NaiveActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super(NaiveActionSpace, self).__init__(env)

        self.action_space = gym.spaces.Discrete(self.env.num_ops)

    def action(self, action):
        return action

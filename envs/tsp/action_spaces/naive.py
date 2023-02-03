import gym


class NaiveActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super(NaiveActionSpace, self).__init__(env)

    def action(self, action):
        return action

import gym


class NaiveObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super(NaiveObservationSpace, self).__init__(env)

    def observation(self, observation):
        return observation

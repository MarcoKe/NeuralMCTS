import gym
import numpy as np

# Parameters previously taken from the param_parser TODO take from arguments
high = 9  # duration upper bound
low = 1  # duration lower bound


class GNNObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super(GNNObservationSpace, self).__init__(env)
        low_bounds = np.tile(np.array([low, 0], dtype=np.float32), (self.env.num_ops, 1))
        high_bounds = np.tile(np.array([high, 1], dtype=np.float32), (self.env.num_ops, 1))
        self.observation_space = gym.spaces.Dict(
            {"adj_matrix": gym.spaces.Box(low=0, high=1, shape=self.env.state['adj_matrix'].shape, dtype=np.float32),
             "features": gym.spaces.Box(low=low_bounds, high=high_bounds,
                                        shape=self.env.state['features'].shape, dtype=np.float32)})

    def observation(self, observation):
        return {"adj_matrix": observation['adj_matrix'], "features": observation['features']}

    def normalize(self, val, min, max):
        return val - min / (max - min)

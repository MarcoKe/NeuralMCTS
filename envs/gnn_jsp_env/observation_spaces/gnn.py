import gym
import numpy as np


class GNNObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super(GNNObservationSpace, self).__init__(env)
        # The observation space consists of the graph's adjacency matrix, concatenated with the node features
        adj_matrix_shape = self.env.state['adj_matrix'].shape
        shape = (adj_matrix_shape[0], adj_matrix_shape[1] + self.env.state['features'].shape[1])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape, np.float32)

    def observation(self, observation):
        node_inf = []
        for i in range(self.env.num_ops):
            node_inf.append(np.append(observation['adj_matrix'][i], observation['features'][i]))
        return np.array(node_inf).astype(np.float32)

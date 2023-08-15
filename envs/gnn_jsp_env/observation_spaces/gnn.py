import gym
import numpy as np


class GNNObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super(GNNObservationSpace, self).__init__(env)
        # The observation space consists of the graph's adjacency matrix, concatenated with the node features
        # and node states (1 if the node corresponds to an action which can be carried out in the current
        # state, 0 if not, and -1 if it corresponds to an action which is last in its job and has already been
        # completed (needed for consistency))
        adj_matrix_shape = self.env.state['adj_matrix'].shape
        shape = (adj_matrix_shape[0], adj_matrix_shape[1] + self.env.state['features'].shape[1] + 1)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape, np.float32)

    def observation(self, observation):
        node_inf = []
        for i in range(self.env.num_ops):
            node_inf.append(np.concatenate((observation['adj_matrix'][i], observation['features'][i],
                                           observation['node_states'][:, np.newaxis][i])))
        return np.array(node_inf).astype(np.float32)

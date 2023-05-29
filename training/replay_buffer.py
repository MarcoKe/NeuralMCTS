import numpy as np
import torch

# there are almost definitely ways to implement a more efficient replay memory
# however, the time required to collect experiences far exceeds the training time
# optimizing this replay buffer should probably not be top priority


class ReplayMemory:
    def __init__(self, obs_dim, probs_dim, size):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.probs_buf = np.zeros((size, probs_dim), dtype=np.float32)
        self.outcomes_buf = np.zeros(size, dtype=np.float32)
        self.children_val_buf = np.zeros((size, probs_dim))
        self.children_obs_buf = np.zeros((size, probs_dim, *obs_dim))
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, probs, outcomes, children_vals, children_obs):
        for o, p, z, cv, co in zip(obs, probs, outcomes, children_vals, children_obs):
            self.store_(o, p, z, cv, co)

    def store_(self, ob, prob, outcome, cval, cobs):
        self.obs_buf[self.ptr] = ob
        self.probs_buf[self.ptr] = prob
        self.outcomes_buf[self.ptr] = outcome
        self.children_obs_buf[self.ptr] = cobs
        self.children_val_buf[self.ptr] = cval

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     mcts_probs=self.probs_buf[idxs],
                     outcomes=self.outcomes_buf[idxs],
                     children_vals=self.children_val_buf[idxs],
                     children_obs=self.children_obs_buf[idxs]
                     )

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def __len__(self):
        return self.size
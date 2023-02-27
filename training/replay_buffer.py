import numpy as np
import torch


class ReplayMemory:
    def __init__(self, obs_dim, probs_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.probs_buf = np.zeros((size, probs_dim), dtype=np.float32)
        self.outcomes_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, probs, outcomes):
        for o, p, z in zip(obs, probs, outcomes):
            self.store_(o, p, z)

    def store_(self, ob, prob, outcome):
        self.obs_buf[self.ptr] = ob
        self.probs_buf[self.ptr] = prob
        self.outcomes_buf[self.ptr] = outcome

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     mcts_probs=self.probs_buf[idxs],
                     outcomes=self.outcomes_buf[idxs],
                     )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def __len__(self):
        return self.size
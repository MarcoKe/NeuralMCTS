import numpy as np
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch as th
from torch import nn


class GNNExtractor(BaseFeaturesExtractor):
    """
    Graph Neural Network feature extractor.

    :param observation_space:
    :param graph_pool: type of pooling used (avg or sum)
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, num_layers: int = 3, num_mlp_layers: int = 2,
                 hidden_dim: int = 64, graph_pool: str = "avg", device: str = "auto"):
        super().__init__(observation_space=observation_space, features_dim=64)

        assert len(observation_space.spaces.keys()) == 2 and list(observation_space.spaces.keys())[0] == "adj_matrix" \
               and list(observation_space.spaces.keys())[1] == "features", (
            "The observation space should consist of a graph adjacency matrix and node features")

        self.graph_pool = graph_pool
        self.device = get_device(device)
        self.num_layers = num_layers
        feature_space = list(observation_space.spaces.values())[1]
        input_dim = feature_space.shape[1]

        # List of MLPs
        self.mlps = th.nn.ModuleList()
        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = th.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.mlps = self.mlps.to(self.device)
        self.batch_norms = self.batch_norms.to(self.device)

    def next_layer(self, h, layer, adj_block=None):
        pooled = th.mm(adj_block, h)
        if self.graph_pool == "avg":
            # If average pooling
            degree = th.mm(adj_block, th.ones((adj_block.shape[0], 1)).to(self.device))
            pooled = pooled / degree
        else:
            raise NotImplementedError()
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        # non-linearity
        h = th.nn.functional.relu(h)
        return h

    def forward(self, observations: th.Tensor) -> th.Tensor:
        adj_block = th.stack([th.from_numpy(np.copy(l)).to_sparse() for l in
                              list(observations.values())[0]])
        n_tasks = adj_block.shape[1]
        adj_block_concat = self.aggr_obs(adj_block, n_tasks)
        fea_concat = th.stack([th.from_numpy(np.copy(l)) for l in list(observations.values())[1]])
        fea_concat = fea_concat.reshape(-1, fea_concat.size(-1))
        h = fea_concat  # list of hidden representations at each layer (including input)

        graph_pool_cal = self.g_pool_cal(self.graph_pool,
                                         batch_size=adj_block.shape,
                                         n_nodes=n_tasks,
                                         device=self.device)

        for layer in range(self.num_layers - 1):
            h = self.next_layer(h, layer, adj_block_concat)

        pooled_h = th.sparse.mm(graph_pool_cal, h)

        return pooled_h

    def g_pool_cal(self, graph_pool_type, batch_size, n_nodes, device):
        # batch_size is the shape of batch
        # for graph pool sparse matrix
        if graph_pool_type == 'avg':
            elem = th.full(size=(batch_size[0] * n_nodes, 1),
                           fill_value=1 / n_nodes,
                           dtype=th.float32,
                           device=device).view(-1)
        else:
            elem = th.full(size=(batch_size[0] * n_nodes, 1),
                           fill_value=1,
                           dtype=th.float32,
                           device=device).view(-1)
        idx_0 = th.arange(start=0, end=batch_size[0], device=device, dtype=th.long)
        idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0] * n_nodes, 1)).squeeze()

        idx_1 = th.arange(start=0, end=n_nodes * batch_size[0], device=device, dtype=th.long)
        idx = th.stack((idx_0, idx_1))
        graph_pool = th.sparse.FloatTensor(idx, elem, th.Size([batch_size[0], n_nodes * batch_size[0]])).to(device)

        return graph_pool

    def aggr_obs(self, obs_mb, n_nodes):
        idxs = obs_mb.coalesce().indices()
        vals = obs_mb.coalesce().values()
        new_idx_row = idxs[1] + idxs[0] * n_nodes
        new_idx_col = idxs[2] + idxs[0] * n_nodes
        idx_mb = th.stack((new_idx_row, new_idx_col))
        adj_batch = th.sparse.FloatTensor(indices=idx_mb,
                                          values=vals,
                                          size=th.Size([obs_mb.shape[0] * n_nodes,
                                                        obs_mb.shape[0] * n_nodes]),
                                          ).to(obs_mb.device)
        return adj_batch


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                    If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        """
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = th.nn.ModuleList()
            self.batch_norms = th.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                h = self.batch_norms[layer](h)
                h = th.nn.functional.relu(h)
            return self.linears[self.num_layers - 1](h)

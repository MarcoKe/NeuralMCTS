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
        super().__init__(observation_space=observation_space, features_dim=5)

        assert len(observation_space.spaces.keys()) == 2 and list(observation_space.spaces.keys())[0] == "adj_matrix" \
               and list(observation_space.spaces.keys())[1] == "features", (
            "The observation space should consist of a graph adjacency matrix and node features"
        )

        self.device = get_device(device)
        self.num_layers = num_layers
        adj_matrix_space = list(observation_space.spaces.values())[0]
        feature_space = list(observation_space.spaces.values())[1]
        self.graph_pool = self.g_pool_cal(graph_pool, batch_size=(1, 1), n_nodes=adj_matrix_space.shape[0],
                                          device=self.device)
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

        # output should be batch size x features_dim!
        self.linear = nn.Linear(hidden_dim, 1).to(self.device)  # 1 should be batch size?

    def next_layer(self, h, layer, adj_block=None):
        pooled = th.mm(adj_block, h)
        if self.graph_pool == "avg":
            # If average pooling
            degree = th.mm(adj_block, th.ones((adj_block.shape[0], 1)).to(self.device))
            pooled = pooled / degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        # non-linearity
        h = th.nn.functional.relu(h)
        return h

    def forward(self, observations: th.Tensor) -> th.Tensor:
        adj_block = list(observations.values())[0][0]  # TODO make compatible with batches of obs
        x_concat = list(observations.values())[1][0]

        # list of hidden representation at each layer (including input)
        h = x_concat

        print("gnn forward")
        print("adj_block shape", adj_block.shape)
        print("x_concat shape", x_concat.shape)

        for layer in range(self.num_layers - 1):
            h = self.next_layer(h, layer, adj_block)

        h_nodes = h.clone()#.reshape(1, -1)  # 4x64 before reshape, 1x256 after
        pooled_h = th.sparse.mm(self.graph_pool, h)  # 1x64

        h_cat = th.cat((pooled_h, h_nodes), dim=0)  # 5x64
        h_cat = th.transpose(self.linear(h_cat), 0, 1)

        # print("gnn forward")
        # print("adj_block shape", adj_block.shape)
        # print("x_concat shape", x_concat.shape)
        print("h_cat shape", h_cat.shape)
        print("len obs", len(list(observations.values())[0]))

        return h_cat  # should be obs.dim x 5

    def g_pool_cal(self, graph_pool_type, batch_size, n_nodes, device):
        # batch_size is the shape of batch
        # for graph pool sparse matrix
        if graph_pool_type == 'average':
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
                h = th.nn.functional.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

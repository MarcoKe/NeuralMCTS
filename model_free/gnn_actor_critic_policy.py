from typing import Callable, Tuple, Type, Optional, List, Union, Dict

from gym import spaces
import torch as th
from torch import nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from stable_baselines3.common.distributions import Distribution, CategoricalDistribution, make_proba_distribution
from sb3_contrib.common.maskable.distributions import MaskableDistribution, MaskableCategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor, get_device


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param net_arch: specification of the policy and value networks
    :param feature_dim_policy: dimension of the node features extracted with the feature extractor
    :param feature_dim_value: dimension of the graph feature extracted with the feature extractor
    """

    def __init__(
            self,
            net_arch: Union[List[int], Dict[str, List[int]]],
            feature_dim_policy: int = 128,
            feature_dim_value: int = 64,
            activation_fn: Type[nn.Module] = nn.Tanh(),
            device: Union[th.device, str] = "auto",
    ):
        super().__init__()

        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        prev_layer_dim_pi = feature_dim_policy
        prev_layer_dim_vf = feature_dim_value

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(prev_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            prev_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(prev_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            prev_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = prev_layer_dim_pi
        self.latent_dim_vf = prev_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features, pooled_h) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(pooled_h)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class GNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):

        # Default network architecture, from L2D
        if net_arch is None:
            net_arch = dict(pi=[32, 1], vf=[32, 1])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = CustomNetwork(net_arch=self.net_arch, activation_fn=self.activation_fn)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, CategoricalDistribution) or \
                isinstance(self.action_dist, MaskableCategoricalDistribution):
            self.action_net = nn.Linear(latent_dim_pi, 1)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        node_features, graph_feature = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(node_features, graph_feature)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        dim0, dim1 = latent_pi.shape[0], latent_pi.shape[1]
        mean_actions = self.action_net(latent_pi).reshape(dim0, dim1)

        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        node_features, graph_feature = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(node_features, graph_feature)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        node_features, graph_feature = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(node_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        node_features, graph_feature = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(graph_feature)
        return self.value_net(latent_vf)

    def obs_to_tensor(self, observation: np.ndarray) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Note: this is highly simplified compared to the original Stable Baselines function,
        only works with the GNN observation space and feature extractor.

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        observation = np.array(observation)
        observation = obs_as_tensor(observation, self.device)
        vectorized_env = True
        return observation, vectorized_env


class MaskableGNNActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):

        # Default network architecture, from L2D
        if net_arch is None:
            net_arch = dict(pi=[32, 1], vf=[32, 1])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = CustomNetwork(net_arch=self.net_arch, activation_fn=self.activation_fn)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, MaskableCategoricalDistribution):
            self.action_net = nn.Linear(latent_dim_pi, 1)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False, action_masks: Optional[np.ndarray] = None) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Applied to the distribution to exclude illegal actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        node_features, graph_feature = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(node_features, graph_feature)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        dim0, dim1 = latent_pi.shape[0], latent_pi.shape[1]
        mean_actions = self.action_net(latent_pi).reshape(dim0, dim1)

        if isinstance(self.action_dist, MaskableCategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, action_masks: Optional[np.ndarray] = None) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :param action_masks:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        node_features, graph_feature = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(node_features, graph_feature)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :param action_masks:
        :return: the action distribution.
        """
        node_features, graph_feature = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(node_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        node_features, graph_feature = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(graph_feature)
        return self.value_net(latent_vf)

    def obs_to_tensor(self, observation: np.ndarray) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Note: this is highly simplified compared to the original Stable Baselines function,
        only works with the GNN observation space and feature extractor.

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        observation = np.array(observation)
        observation = obs_as_tensor(observation, self.device)
        vectorized_env = True
        return observation, vectorized_env


if __name__ == '__main__':
    from experiment_management.config_handling.load_exp_config import load_yml
    from experiment_management.setup_experiment import create_env
    from pathlib import Path
    from gnn_feature_extractor import GNNExtractor

    env_config = load_yml((Path('../data/config/envs/jsp_gnn_001.yml')))
    env, _, _ = create_env(env_config)

    feature_extractor_kwargs = dict(num_layers=3, num_mlp_layers=2, input_dim=2,
                                    hidden_dim=64, graph_pool="avg")
    policy_kwargs = dict(activation_fn=th.nn.modules.activation.ReLU,
                         net_arch=[dict(pi=[32], vf=[32])],
                         features_extractor_class=GNNExtractor,
                         features_extractor_kwargs=feature_extractor_kwargs)

    model = PPO(GNNActorCriticPolicy, env, verbose=1, learning_rate=0.00005,
                policy_kwargs=policy_kwargs)
    model.learn(5000)

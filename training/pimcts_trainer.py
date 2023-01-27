import stable_baselines3.common.utils
from stable_baselines3 import PPO
import torch as th
from torch.nn import functional as F
from envs.TSP import TSPGym, TSP
import gym
from stable_baselines3.common.utils import explained_variance
import numpy as np
import random
from stable_baselines3.common.policies import ActorCriticPolicy
from mcts.mcts_main import MCTSAgent
from scipy.stats import entropy


class MCTSPolicyImprovementTrainer:
    def __init__(self, env, mcts_agent: MCTSAgent, model_free_agent):
        """
        :mcts_agent: is the agent generating experiences by performing mcts searches
        :model_free_agent: is the model free agent that is being trained using these collected experiences
        typically, the policy and/or value function of the model free agent are also a part of the mcts agent
        """
        self.env = env
        self.model_free_agent = model_free_agent
        self.policy: ActorCriticPolicy = model_free_agent.policy
        self.mcts_agent = mcts_agent
        self.model_free_agent.learn(total_timesteps=1)
        self.policy.optimizer.weight_decay = 0.03
        self.policy.optimizer.learning_rate = 1e-5

    def mcts_policy_improvement_loss(self, pi_mcts, pi_theta, v_mcts, v_theta):
        """
        cross entropy between model free policy prediction and mcts policy
        mse between value estimates
        """
        policy_loss = th.mean(-th.sum(pi_mcts * th.log(pi_theta), dim=-1))
        value_loss = F.mse_loss(v_mcts, v_theta) / 5
        print("ce loss: ", policy_loss, " mse: ", value_loss)
        total_loss = (policy_loss + value_loss) / 2 # + todo regularization
        return total_loss, policy_loss, value_loss

    def forward(self, obs: th.Tensor):
        """
        Modification of stb3 forward pass so that probabilities for all actions are computed and returned
        """
        # Preprocess the observation if needed
        features = self.policy.extract_features(obs)
        if self.policy.share_features_extractor:
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.policy.value_net(latent_vf)
        action_logits = self.policy.action_net(latent_pi)
        action_probs = th.nn.functional.softmax(action_logits, dim=1)
        return action_probs, values

    def train_on_mcts_experiences(self, observations: th.Tensor, mcts_probs: th.Tensor,
              mcts_values: th.Tensor) -> None:
        """
        Update policy using the results from mcts searches.
        :observations: [num_experiences, obs_size]
        :legal_actions: [num_experiences, num_actions]
        :mcts_probs: [num_experiences, num_actions]
        :mcts_values: [num_experiences, 1]
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # train for n_epochs epochs
        # batches
        predicted_probs, predicted_values = self.forward(observations) # legal actions or just all actions?


        loss, ploss, vloss = self.mcts_policy_improvement_loss(mcts_probs, predicted_probs, mcts_values, predicted_values)

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()
        #
        # # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        #
        print("loss: ", loss.item())
        # Logs
        # self.logger.record("mctstrain/entropy_loss", np.mean(entropy_losses))
        self.model_free_agent.logger.record("mctstrain/policy_ce_loss", ploss.item())
        self.model_free_agent.logger.record("mctstrain/value_mse_loss", vloss.item())
        self.model_free_agent.logger.record("mctstrain/mcts_probs_entropy", np.mean(entropy(mcts_probs.detach().numpy(), base=model.max_num_actions(), axis=1)))
        self.model_free_agent.logger.record("mctstrain/learned_probs_entropy", np.mean(entropy(predicted_probs.detach().numpy(), base=model.max_num_actions(), axis=1)))

        # self.logger.record("mctstrain/clip_fraction", np.mean(clip_fractions))
        # self.logger.record("mctstrain/loss", loss.item())
        # # self.logger.record("mctstrain/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("mctstrain/std", th.exp(self.policy.log_std).mean().item())
        # log policy entropy
        self.policy.set_training_mode(False)

    def collect_experience(self, num_episodes=5):
        pi_mcts = []
        v_mcts = []
        observations = []
        rewards_list = []
        rewards = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False

            num_steps = 0
            while not done:
                pi_mcts_, v_mcts_, action = self.mcts_agent.stochastic_policy(self.env.raw_state())
                observations.append(state)
                pi_mcts.append(pi_mcts_.tolist())
                v_mcts.append(v_mcts_)
                state, reward, done, _ = self.env.step(action)
                num_steps += 1

            rewards_list.extend([reward] * num_steps)
            rewards += reward
        print("avg reward: ", rewards / num_episodes)
        v_mcts = rewards_list
        return observations, pi_mcts, v_mcts, rewards / num_episodes

    def train(self, policy_improvement_iterations=100):
        for _ in range(policy_improvement_iterations):
            print("collecting experience")
            observations, pi_mcts, v_mcts = self.collect_experience()
            observations = th.Tensor(observations)
            pi_mcts = th.Tensor(pi_mcts)
            v_mcts = th.Tensor(v_mcts)
            print("training", observations.shape, pi_mcts.shape, v_mcts.shape)
            self.train_on_mcts_experiences(observations, pi_mcts, v_mcts)

    def train_parallel(self, policy_improvement_iterations=1000):
        for i in range(policy_improvement_iterations):
            print("collecting experience")
            import multiprocess as mp

            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap(self.collect_experience, [[]] * 8)



            pool.close()
            for r in results:
                observations = th.Tensor(r[0])
                pi_mcts = th.Tensor(r[1])
                v_mcts = th.Tensor(r[2])
                avg_reward = r[3]
                print("training", observations.shape, pi_mcts.shape, v_mcts.shape)
                self.train_on_mcts_experiences(observations, pi_mcts, v_mcts)
                self.model_free_agent.logger.record('mctstrain/ep_rew', avg_reward)

            self.model_free_agent.logger.record('mctstrain/policy_improvement_iter', i)

            self.model_free_agent.logger.dump(step=i)


if __name__ == '__main__':
    num_cities = 15
    env = TSPGym(num_cities=num_cities)
    model = TSP(num_cities=15)

    # agent = PPO.load("ppo_tsp_15_1e6.zip")
    policy_kwargs = dict(activation_fn=th.nn.modules.activation.Mish)
    # model_free_agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log="stb3_tsp_tensorboard/", policy_kwargs=policy_kwargs)
    model_free_agent = PPO.load('results/trained_agents/tsp/model_free/ppo_tsp_15_1e6_ent.zip', env=env)
    from mcts.tree_policies.tree_policy import UCTPolicy
    from mcts.tree_policies.exploration_terms.puct_term import PUCTTerm
    from mcts.tree_policies.exploitation_terms.avg_node_value import AvgNodeValueTerm
    from mcts.expansion_policies.expansion_policy import ExpansionPolicy
    from mcts.evaluation_policies.neural_value_eval import NeuralValueEvalPolicy
    from mcts.evaluation_policies.neural_rollout_policy import NeuralRolloutPolicy
    from model_free.stb3_wrapper import Stb3ACAgent


    tp = UCTPolicy(AvgNodeValueTerm(), PUCTTerm(exploration_constant=1))
    ep = ExpansionPolicy(model=model)
    rp = NeuralRolloutPolicy(model_free_agent=Stb3ACAgent(model_free_agent), model=model)
    mcts_agent = MCTSAgent(model, tp, ep, rp, num_simulations=100)

    trainer = MCTSPolicyImprovementTrainer(env, mcts_agent, model_free_agent)
    trainer.train_parallel()

    model_free_agent.save('results/trained_agents/tsp/nmcts/pimcts_15')
    #
    # num_experiences = 128
    #
    # obs = []
    # mcts_probs = []
    # mcts_values = []
    # for _ in range(num_experiences):
    #     obs.append(env.reset())
    #     mcts_values.append(random.random())
    #
    # mcts_probs = np.random.dirichlet(np.ones(num_cities), size=num_experiences)
    #
    # obs_tensor = th.Tensor(obs)
    # mcts_probs_tensor = th.Tensor(mcts_probs)
    # mcts_values_tensor = th.Tensor(mcts_values)
    # mcts_values_tensor = mcts_values_tensor[:, None]
    #
    # print("obs size", obs_tensor.shape)
    # print("probs size", mcts_probs_tensor.shape)
    # print("values size", mcts_values_tensor.shape)
    # # agent = PPO.load("ppo_tsp_15_1e6.zip")
    # # print(agent.get_parameters())
    # # print(agent.policy)
    #
    # # model = MCTSActorCriticPolicy(env.observation_space, env.action_space, stable_baselines3.common.utils.constant_fn(0.3))
    # # model.load_state_dict(th.load("ppo_tsp_15_1e6/policy.pth"))
    # agent = PPO.load("ppo_tsp_15_1e6.zip")
    # trainer = MCTSPolicyImprovementTrainer(agent.policy)
    #
    #
    # trainer.train_on_mcts_experiences(obs_tensor, mcts_probs_tensor, mcts_values_tensor)

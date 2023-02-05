from stable_baselines3 import PPO
import torch as th
from torch.nn import functional as F
from envs.tsp.TSP import TSPGym, TSP
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from mcts.mcts_agent import MCTSAgent
from scipy.stats import entropy
import multiprocess as mp
import wandb
from mcts.util.benchmark_agents import perform_episode as eval
from mcts.util.benchmark_agents import opt_gap
import copy
from envs.tsp.tsp_solver import TSPSolver
from mcts.util.benchmark_agents import MCTSAgentWrapper
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


class MCTSPolicyImprovementTrainer:
    def __init__(self, env, mcts_agent: MCTSAgent, model_free_agent, weight_decay=0.0005, learning_rate=1e-5,
                 buffer_size=50000, batch_size=256, num_epochs=1, policy_improvement_iterations=2000, workers=8,
                 num_episodes=5, wandb_run=None):
        """
        :mcts_agent: is the agent generating experiences by performing mcts searches
        :model_free_agent: is the model free agent that is being trained using these collected experiences
                     typically, the policy and/or value function of the model free agent are also a part of the mcts agent
        :batch_size: batches of this size will be sampled from the replay buffer during training
        :num_epochs: after experience has been collected and deposited to the replay buffer, we will train for this
                     number of epochs on the data in the buffer
        :policy_improvement_iterations: number of (collect experience) -> (train) iterations
        :workers: how many workers will collect experience in parallel
        :num_episodes: how many episodes of experience EACH worker will collect
        """
        self.env = env
        self.model_free_agent = model_free_agent
        self.policy: ActorCriticPolicy = model_free_agent.policy
        self.mcts_agent = mcts_agent
        self.policy.optimizer.weight_decay = weight_decay
        self.policy.optimizer.learning_rate = learning_rate
        self.wandb_run = wandb_run
        self.memory = ReplayMemory(env.observation_space.shape[0], env.max_num_actions(), buffer_size)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.policy_improvement_iterations = policy_improvement_iterations
        self.workers = workers
        self.num_episodes = num_episodes

        if not self.wandb_run:
            self.model_free_agent.learn(total_timesteps=1)

    def log(self, key, value):
        if self.wandb_run:
            self.wandb_run.log({key: value, 'policy_improvement_steps': self.policy_improvement_steps})
        else:
            self.model_free_agent.logger.record(key, value)

    @staticmethod
    def mcts_policy_improvement_loss(pi_mcts, pi_theta, v_mcts, v_theta):
        """
        cross entropy between model free policy prediction and mcts policy
        mse between value estimates
        """
        policy_loss = th.mean(-th.sum(pi_mcts * th.log(pi_theta), dim=-1))
        value_loss = F.mse_loss(v_mcts, v_theta) / 2 #todo
        total_loss = (policy_loss + value_loss) / 2
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

    def train_on_mcts_experiences(self) -> None:
        """
        Update policy using the results from mcts searches.
        :observations: [num_experiences, obs_size]
        :legal_actions: [num_experiences, num_actions]
        :mcts_probs: [num_experiences, num_actions]
        :mcts_values: [num_experiences, 1]
        """

        if len(self.memory) < self.batch_size: return

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        for training_iter in range(self.num_epochs * int(len(self.memory) / self.batch_size)):
            print("training, batch", training_iter, " size memory buffer", len(self.memory), " / ", self.memory.max_size)
            batch = self.memory.sample_batch(self.batch_size)

            predicted_probs, predicted_values = self.forward(batch['obs']) # legal actions or just all actions?

            loss, ploss, vloss = self.mcts_policy_improvement_loss(batch['mcts_probs'], predicted_probs, batch['outcomes'], predicted_values)

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

            self.log("mctstrain/policy_ce_loss", ploss.item())
            self.log("mctstrain/value_mse_loss", vloss.item())
            self.log("mctstrain/mcts_probs_entropy", np.mean(entropy(batch['mcts_probs'].detach().numpy(), base=self.mcts_agent.env.max_num_actions(), axis=1)))
            self.log("mctstrain/learned_probs_entropy", np.mean(entropy(predicted_probs.detach().numpy(), base=self.mcts_agent.env.max_num_actions(), axis=1)))

        self.policy.set_training_mode(False)

    def perform_episode(self):
        state = self.env.reset()
        done = False

        observations = []
        pi_mcts = []
        v_mcts = []
        num_steps = 0

        while not done:
            pi_mcts_, v_mcts_, action = self.mcts_agent.stochastic_policy(self.env.raw_state())
            observations.append(state)
            pi_mcts.append(pi_mcts_.tolist())
            v_mcts.append(v_mcts_)
            state, reward, done, _ = self.env.step(action)
            num_steps += 1

        return observations, pi_mcts, v_mcts, num_steps, reward

    def collect_experience(self):
        pi_mcts = []
        v_mcts = []
        observations = []
        rewards_list = []
        rewards = 0
        for ep in range(self.num_episodes):
            o_, pi_mcts_, v_mcts_, num_steps, reward = self.perform_episode()
            observations.extend(o_)
            pi_mcts.extend(pi_mcts_)
            v_mcts.extend(v_mcts_)
            rewards += reward
            rewards_list.extend([reward] * num_steps)

        return observations, pi_mcts, rewards_list, rewards / self.num_episodes

    def train(self):
        for i in range(self.policy_improvement_iterations):
            self.policy_improvement_steps = i
            print("collecting experience")

            if self.workers > 1:
                pool = mp.Pool(self.workers)
                results = pool.starmap(self.collect_experience, [[]] * self.workers)
                pool.close()
            else:
                results = [self.collect_experience()]

            for r in results:
                observations = th.Tensor(r[0]) #todo: we are converting between different datastructures in this file. is all of it necessary?
                pi_mcts = th.Tensor(r[1])
                outcomes = th.Tensor(r[2])
                avg_reward = r[3]
                self.memory.store(observations, pi_mcts, outcomes)
                self.log('mctstrain/ep_rew', avg_reward)

            self.train_on_mcts_experiences()

            self.log('mctstrain/policy_improvement_iter', i)

            self.evaluate()
            if not self.wandb_run:
                self.model_free_agent.logger.dump(step=i)

        model_free_agent.save('results/trained_agents/tsp/nmcts/pimcts_15') #todo automatically name according to experimetn

    def evaluate(self, eval_iterations=10):
        opt_gaps = 0
        for _ in range(eval_iterations):
            state = copy.deepcopy(self.env.reset())
            state_ = copy.deepcopy(self.env.raw_state())
            opt = TSPSolver.solve(state_)
            reward = eval(self.env, MCTSAgentWrapper(self.mcts_agent, self.env), state_, state)
            opt_gaps += opt_gap(opt, -reward)

        self.log('mctstrain/eval_optgap', opt_gaps / eval_iterations)


if __name__ == '__main__':
    wandb.require("service")

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
    from mcts.evaluation_policies.neural_rollout_policy import NeuralRolloutPolicy
    from model_free.stb3_wrapper import Stb3ACAgent


    tp = UCTPolicy(AvgNodeValueTerm(), PUCTTerm(exploration_constant=1))
    ep = ExpansionPolicy()
    rp = NeuralRolloutPolicy()
    mcts_agent = MCTSAgent(model, tp, ep, rp, neural_net=Stb3ACAgent(model_free_agent), num_simulations=100)

    trainer = MCTSPolicyImprovementTrainer(env, mcts_agent, model_free_agent)
    trainer.train()

    # model_free_agent.save('results/trained_agents/tsp/nmcts/pimcts_15')
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

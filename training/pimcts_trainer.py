import copy
import torch as th
import numpy as np
from scipy.stats import entropy
import multiprocess as mp
import wandb
import os
import time
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

from envs.tsp.TSP import TSPGym, TSP
from mcts.mcts_agent import MCTSAgent
from mcts.util.benchmark_agents import perform_episode as eval
from mcts.util.benchmark_agents import opt_gap
from mcts.util.benchmark_agents import MCTSAgentWrapper, Stb3AgentWrapper
from training.replay_buffer import ReplayMemory
from training.schedule import LinearSchedule

class MCTSPolicyImprovementTrainer:
    def __init__(self, exp_name, env, mcts_agent: MCTSAgent, model_free_agent, weight_decay=0.0005, learning_rate=1e-5,
                 buffer_size=50000, batch_size=256, num_epochs=1, policy_improvement_iterations=2000, workers=8,
                 num_episodes=5, solver=None, wandb_run=None):
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
        self.exp_name = exp_name
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
        self.solver = solver

        if not self.wandb_run:
            self.model_free_agent.learn(total_timesteps=1)

        self.temp_schedule = LinearSchedule(0.9, 1.0, self.policy_improvement_iterations)

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
            temp = self.temp_schedule.value(self.policy_improvement_steps)
            self.log('mctstrain/temp', temp)
            pi_mcts_, v_mcts_, action = self.mcts_agent.stochastic_policy(self.env.raw_state(), temperature=temp)
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
            start_time = time.time()

            if self.workers > 1:
                pool = mp.Pool(self.workers)
                results = pool.starmap(self.collect_experience, [[]] * self.workers)
                pool.close()
            else:
                results = [self.collect_experience()]

            for r in results:
                observations = th.Tensor(np.array(r[0])) #todo: we are converting between different datastructures in this file. is all of it necessary?
                pi_mcts = th.Tensor(r[1])
                outcomes = th.Tensor(r[2])
                avg_reward = r[3]
                self.memory.store(observations, pi_mcts, outcomes)
                self.log('mctstrain/ep_rew', avg_reward)
                self.log('time/collecting', time.time() - start_time)

            start_time = time.time()
            self.train_on_mcts_experiences()

            self.log('mctstrain/policy_improvement_iter', i)
            self.log('time/training', time.time()-start_time)
            start_time = time.time()
            if i % 1 == 0:
                self.evaluate()
            self.log('time/evaluation', time.time() - start_time)
            if not self.wandb_run:
                self.model_free_agent.logger.dump(step=i)

        model_path = 'results/trained_agents/' + self.exp_name
        if self.wandb_run:
            model_path = os.path.join(self.wandb_run.dir, self.exp_name)
        self.model_free_agent.save(model_path)

    def evaluate(self, eval_iterations=1):
        if self.workers > 1:
            pool = mp.Pool(eval_iterations)
            results = pool.starmap(self.evaluate_single, [[]] * eval_iterations)
            pool.close()
        else:
            results = [self.evaluate_single()]

        opt_gaps = 0
        reward_diffs = 0
        for r in results:
            opt_gaps += r[0]
            self.log('eval/opt_gap', r[0])
            self.log('eval/diff_mcts_model_free', r[1])
            reward_diffs += r[1]


        self.log('mctstrain/eval_optgap', opt_gaps / eval_iterations)
        self.log('mctstrain/diff_mcts_model_free', reward_diffs / eval_iterations)

    def evaluate_single(self):
        state = copy.deepcopy(self.env.reset())
        state_ = copy.deepcopy(self.env.raw_state())

        reward_model_free = eval(self.env, Stb3AgentWrapper(self.model_free_agent, self.env, self.mcts_agent.model), copy.deepcopy(state_), copy.deepcopy(state))

        reward_mcts = eval(self.env, MCTSAgentWrapper(self.mcts_agent, self.env), state_, state)
        opt = self.solver.solve(self.env.current_instance())

        gap = opt_gap(opt, -reward_mcts)
        reward_diff = reward_mcts - reward_model_free

        return gap, reward_diff



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

import copy
import torch as th
import numpy as np
from scipy.stats import entropy
import multiprocess as mp

import os
import time
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

from mcts.mcts_agent import MCTSAgent
from mcts.tree_policies.tree_policy_factory import tree_policy_factory
from mcts.expansion_policies.expansion_policy_factory import expansion_policy_factory
from mcts.evaluation_policies.eval_policy_factory import eval_policy_factory
from mcts.util.benchmark_agents import opt_gap
from mcts.util.benchmark_agents import MCTSAgentWrapper, Stb3AgentWrapper
from training.replay_buffer import ReplayMemory
from training.schedule import LinearSchedule
from mcts.node import Node


class MCTSPolicyImprovementTrainer:
    def __init__(self, exp_name, env, eval_env, mcts_agent: MCTSAgent, model_free_agent, weight_decay=0.0005,
                 learning_rate=1e-5,
                 buffer_size=50000, batch_size=256, num_epochs=1, policy_improvement_iterations=2000, workers=8,
                 num_episodes=5, warmup_steps=0, entropy_loss=False, children_value_targets=False,
                 selection_mode='mean', stochastic_actions=False, reuse_root=False,
                 value_loss_weight=1.0, entropy_loss_weight=1.0,
                 solver=None, wandb_run=None):
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
        :warmup_steps: How many policy iteration steps are performed using vanilla MCTS before switching to guided mcts
        """
        self.exp_name = exp_name
        self.env = env
        self.eval_env = eval_env
        self.model_free_agent = model_free_agent
        self.policy: ActorCriticPolicy = model_free_agent.policy
        self.guided_mcts_agent = mcts_agent
        self.mcts_agent = self.build_warmup_agent()  # initialized as vanilla mcts agent, replaced by guided agent after # warmup steps
        self.policy.optimizer.weight_decay = weight_decay
        self.policy.optimizer.learning_rate = learning_rate
        self.wandb_run = wandb_run
        self.memory = ReplayMemory(env.observation_space.shape,
                                   env.max_num_actions(), buffer_size)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.policy_improvement_iterations = policy_improvement_iterations
        self.total_model_steps = 0
        self.children_value_targets = children_value_targets
        self.total_neural_net_calls = 0
        self.workers = workers
        self.num_episodes = num_episodes
        self.warmup_steps = warmup_steps
        self.entropy_loss = entropy_loss
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.stochastic_actions = stochastic_actions
        self.reuse_root = reuse_root
        self.selection_mode = selection_mode
        self.solver = solver

        if not self.wandb_run:
            self.model_free_agent.learn(total_timesteps=1)

        self.temp_schedule = LinearSchedule(0.9, 1.0, self.policy_improvement_iterations)

    def log(self, key, value):
        if self.wandb_run:
            self.wandb_run.log({key: value, 'policy_improvement_steps': self.policy_improvement_steps})
        else:
            self.model_free_agent.logger.record(key, value)

    def build_warmup_agent(self):
        """
        This method simply returns a vanilla MCTS agent without neural guidance. The experience collected with this agent
        will still be used for neural net training, however.
        """
        tree_policy = tree_policy_factory.get('uct', **{'exploitation': {'name': 'avg_node_value', 'params': {}},
                                                        'exploration': {'name': 'uct',
                                                                        'params': {'exploration_constant': 1}}})
        expansion_policy = expansion_policy_factory.get('full_expansion')
        evaluation_policy = eval_policy_factory.get('random')

        return MCTSAgent(self.guided_mcts_agent.env,
                         self.guided_mcts_agent.model,
                         tree_policy,
                         expansion_policy,
                         evaluation_policy,
                         neural_net=self.guided_mcts_agent.neural_net,
                         num_simulations=self.guided_mcts_agent.num_simulations,
                         # evaluate_leaf_children=self.guided_mcts_agent.evaluate_leaf_children,
                         evaluate_leaf_children=True,
                         value_initialization=False,
                         initialize_tree=False)

    def mcts_policy_improvement_loss(self, pi_mcts, pi_theta, v_mcts, v_theta, c_v_mcts, c_v_theta):
        """
        cross entropy between model free policy prediction and mcts policy
        mse between value estimates
        """
        policy_loss = th.mean(-th.sum(pi_mcts * th.log(pi_theta + 1e-9), dim=-1))
        value_loss = F.mse_loss(v_mcts, v_theta)
        if self.children_value_targets:
            children_value_loss = F.mse_loss(c_v_mcts, c_v_theta)
            value_loss += children_value_loss

        entropy = th.Tensor([0])
        if self.entropy_loss:
            # entropy with base=num_actions. since torch does not allow specifying custom bases, we use the change of base formula
            entropy = -th.mean(th.sum(
                -(pi_theta * th.log(pi_theta) / th.log(th.ones_like(pi_theta) * self.mcts_agent.env.max_num_actions())),
                axis=1))

        policy_loss_weight = 1.0 - self.value_loss_weight - self.entropy_loss_weight
        total_loss = (policy_loss_weight * policy_loss + self.value_loss_weight * value_loss + (self.entropy_loss_weight * entropy)) / 2
        return total_loss, policy_loss, value_loss, entropy

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
            print("training, batch", training_iter, " size memory buffer", len(self.memory), " / ",
                  self.memory.max_size)
            batch = self.memory.sample_batch(self.batch_size)

            # generate network predictions for loss function
            # predicted policy probs and state values for each sampled obs
            predicted_probs, predicted_values = self.forward(batch['obs'])  # legal actions or just all actions?

            # same targets but for the children of the root node at this obs
            predicted_children_values = [self.forward(batch['children_obs'][:, i, :])[1] for i in
                                         range(batch['children_obs'].shape[1])]
            predicted_children_values = th.stack(predicted_children_values, dim=1).squeeze()

            loss, ploss, vloss, eloss = self.mcts_policy_improvement_loss(batch['mcts_probs'], predicted_probs,
                                                                          batch['outcomes'], predicted_values,
                                                                          batch['children_vals'],
                                                                          predicted_children_values)

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

            self.log("mctstrain/policy_ce_loss", ploss.item())
            self.log("mctstrain/value_mse_loss", vloss.item())
            self.log("mctstrain/entropy_loss", eloss.item())

            self.log("mctstrain/mcts_probs_entropy", np.mean(
                entropy(batch['mcts_probs'].detach().numpy(), base=self.mcts_agent.env.max_num_actions(), axis=1)))
            self.log("mctstrain/learned_probs_entropy", np.mean(
                entropy(predicted_probs.detach().numpy(), base=self.mcts_agent.env.max_num_actions(), axis=1)))

        self.policy.set_training_mode(False)

    def perform_episode(self):
        """
        performs a single episode using the nmcts agent
        @return: data collected from the episode
        """
        state = self.env.reset()
        done = False

        observations = []
        pi_mcts = []
        v_mcts = []
        children_v_mcts = []  # the value of each child of the root node as estimated by mcts
        children_observations = []
        num_steps = 0
        model_steps = 0
        neural_net_calls = 0
        node = None

        while not done:
            pi_mcts_, v_mcts_, action, stats, children_observations_, children_v_mcts_, node = \
                self.mcts_agent.stochastic_policy(self.env.raw_state(), temperature=self.temp,
                                                  selection_mode=self.selection_mode,
                                                  exploration=self.stochastic_actions, root=node)
            observations.append(state)
            pi_mcts.append(pi_mcts_.tolist())
            v_mcts.append(v_mcts_)
            children_v_mcts.append(children_v_mcts_)
            children_observations.append(children_observations_)
            state, reward, done, _ = self.env.step(action)
            if self.reuse_root:
                node = Node.create_root(node, action)
            else:
                node = None

            num_steps += 1
            model_steps += stats['model_steps']
            neural_net_calls += stats['neural_net_calls']

        return observations, pi_mcts, v_mcts, children_v_mcts, children_observations, num_steps, reward, model_steps, \
               neural_net_calls

    def collect_experience(self):
        """
        collects num_episodes of experience by applying nmcts
        """

        pi_mcts = []
        v_mcts = []  # mcts root node value
        children_v_mcts = []  # values of mcts root children
        children_observations = []
        observations = []
        rewards_list = []
        rewards = 0
        model_steps_sum = 0
        neural_net_calls_sum = 0
        for ep in range(self.num_episodes):
            o_, pi_mcts_, v_mcts_, children_v_mcts_, children_obs_, num_steps, reward, model_steps, neural_net_calls = \
                self.perform_episode()
            observations.extend(o_)
            pi_mcts.extend(pi_mcts_)
            children_v_mcts.extend(children_v_mcts_)
            children_observations.extend(children_obs_)
            rewards += reward
            rewards_list.extend([reward] * num_steps)
            model_steps_sum += model_steps
            neural_net_calls_sum += neural_net_calls

        return observations, pi_mcts, rewards_list, rewards / self.num_episodes, model_steps_sum, neural_net_calls_sum, \
               children_v_mcts, children_observations

    def train(self):
        """
        Main training loop consisting of three steps: collecting experience (in parallel if multiple workers),
        storing it in the replay buffer, training neural network from replay buffer, evaluate current state of agent
        """

        for i in range(self.policy_improvement_iterations):
            if i == self.warmup_steps:
                self.mcts_agent = self.guided_mcts_agent
            print(i, " ", self.mcts_agent)
            self.policy_improvement_steps = i
            self.temp = self.temp_schedule.value(self.policy_improvement_steps)
            self.log('mctstrain/temp', self.temp)
            print("collecting experience")
            start_time = time.time()

            if self.workers > 1:
                pool = mp.Pool(self.workers)
                results = pool.starmap(self.collect_experience, [[]] * self.workers)
                pool.close()
            else:
                results = [self.collect_experience()]

            for r in results:
                observations = th.Tensor(np.array(r[0]))  # todo: we are converting between different datastructures in this file. is all of it necessary?
                pi_mcts = th.Tensor(r[1])
                outcomes = th.Tensor(r[2])
                avg_reward = r[3]
                self.total_model_steps += r[4]
                self.total_neural_net_calls += r[5]
                children_v_mcts = r[6]
                children_v_mcts = th.Tensor(children_v_mcts)
                children_observations = th.Tensor(np.array(r[7]))
                self.memory.store(observations, pi_mcts, outcomes, children_v_mcts, children_observations)
                self.log('mctstrain/ep_rew', avg_reward)

            self.log('mctstrain/model_steps', self.total_model_steps)
            self.log('mctstrain/neural_net_calls', self.total_neural_net_calls)

            self.log('time/collecting', time.time() - start_time)

            start_time = time.time()
            self.train_on_mcts_experiences()

            self.log('mctstrain/policy_improvement_iter', i)
            self.log('time/training', time.time() - start_time)
            start_time = time.time()
            if i % 10 == 0:  # todo configure eval frequency
                self.save_model(str(i) + '_' + self.exp_name)
                self.evaluate()
            self.log('time/evaluation', time.time() - start_time)
            if not self.wandb_run:
                self.model_free_agent.logger.dump(step=i)

        self.save_model('final_' + self.exp_name)


    def save_model(self, name):
        model_path = 'results/trained_agents/' + name
        if self.wandb_run:
            model_path = os.path.join(self.wandb_run.dir, name)
        self.model_free_agent.save(model_path)

    def evaluate(self, eval_iterations=8):
        """
        Performs an evaluation of the current agent on instances provided by the eval_env instance generator specified
        in the environment config file. If multiple workers, the evaluation is executed in parallel.
        @param eval_iterations: number of instances to be evaluated.
        """

        if self.workers > 1:
            instances = [(self.eval_env.generator.generate(),) for _ in range(eval_iterations)]
            pool = mp.Pool(eval_iterations)
            results = pool.starmap(self.evaluate_single, instances)
            pool.close()
        else:  # only for debugging purposes
            results = [self.evaluate_single(self.eval_env.generator.generate())]

        for r in results:
            for sol in r[0]:
                self.log('eval/' + sol, r[0][sol])
            self.log('eval/diff_mcts_model_free', r[1])
            self.log('eval/rew_mcts', r[2])
            self.log('eval/rew_learned_policy', r[3])
            self.log('eval/instance_id', r[4])

    def evaluate_single(self, instance):
        """
        Evaluation on a single problem instance using multiple methods (problem-specific solver, model-free, mcts)
        @param instance: the instance to be evaluated on
        @return: optimality gaps of all methods, difference in rewards between model free and mcts, mcts reward,
                 model free reward, instance id
        """

        # optimum and other non-RL methods / heuristics
        solutions = self.solver.solve(copy.deepcopy(instance))
        solution_gaps = self.solutions_to_gaps(solutions)

        # model free and mcts
        eval_env_ = copy.deepcopy(self.eval_env)
        self.mcts_agent.env = eval_env_

        reward_model_free = self.perform_eval_episode(eval_env_, Stb3AgentWrapper(self.model_free_agent, eval_env_,
                                                                                  self.mcts_agent.model),
                                                      copy.deepcopy(instance))

        reward_mcts = self.perform_eval_episode(eval_env_, MCTSAgentWrapper(self.mcts_agent, eval_env_),
                                                copy.deepcopy(instance))
        reward_diff = reward_mcts - reward_model_free
        self.mcts_agent.env = self.env
        return solution_gaps, reward_diff, reward_mcts, reward_model_free, eval_env_.instance.id

    def perform_eval_episode(self, env, agent, instance):
        """
        Performs one evaluation episode by setting a specific problem instance in the environment
        """

        state = env.set_instance(instance)
        state = env.observation(state)
        done = False

        steps = 0
        node = None
        while not done:
            action, node = agent.select_action(state, node)
            state, reward, done, _ = env.step(action)
            node = Node.create_root(node, action)
            steps += 1

        return reward

    def solutions_to_gaps(self, solutions):
        """
        compares different objective values to the optimum and computes corresponding optimality gaps
        @param solutions: dictionary with solution names as keys and objective values as values. optimum should have
                          key 'opt'
        @return: dictionary containing the optimality gaps for every solution
        """

        if 'opt' in solutions:
            opt = solutions['opt']
            gaps = dict()

            for sol in solutions.keys():
                if sol != 'opt':
                    gaps[sol] = opt_gap(opt, solutions[sol])

            return gaps

        else:
            return dict()

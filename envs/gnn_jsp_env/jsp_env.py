from envs.gnn_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator
from envs.gnn_jsp_env.jsp_model import GNNJobShopModel
import gym
import numpy as np
from copy import deepcopy


class GNNJobShopEnv(gym.Env):
    def __init__(self, instance_generator: JSPGenerator, **kwargs):
        self.jsp_generator = instance_generator
        self.model = GNNJobShopModel()

        self.reset()

    def _generate_instance(self):
        self.instance = self.jsp_generator.generate()
        self.ops_per_job = self.instance.num_ops_per_job
        self.num_machines = self.instance.num_machines
        self.max_op_duration = self.instance.max_op_time
        self.num_jobs = self.instance.num_jobs
        self.num_ops = self.num_jobs * self.ops_per_job

        # number of operations scheduled on each machine
        ops_per_machine = [len([op for job in self.instance.jobs for op in job if op.machine_type == m]) for m in
                           range(self.num_machines)]
        # information for each machine: the ids of the operations scheduled on it (in the scheduled order), and the
        # corresponding start and end times
        machine_infos = {m: {'op_ids': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'start_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'end_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32)} for m in range(self.num_machines)}
        # time at which the last scheduled operation ends for each job
        last_job_ops = [-1 for _ in range(self.num_jobs)]
        # time at which the last scheduled operation ends on each machine
        last_machine_ops = [-1 for _ in range(self.num_machines)]

        adj_matrix = self.model.init_adj_matrix(self.num_ops, self.num_jobs)
        features = self.model.init_features(self.instance.jobs)
        schedule = [[] for _ in range(self.num_machines)]
        remaining_ops = [job for job in deepcopy(self.instance.jobs)]

        return {'remaining_ops': remaining_ops, 'schedule': schedule, 'machine_infos': machine_infos,
                'last_job_ops': last_job_ops, 'last_mch_ops': last_machine_ops, 'adj_matrix': adj_matrix,
                'features': features, 'jobs': self.instance.jobs}

    def reset(self):
        self.done = False
        self.state = self._generate_instance()

        return self.state

    def set_state(self, state):
        self.state = state
        if len(state['remaining_ops']) > 0:
            self.done = False

    def step(self, action):
        self.state, reward, self.done = self.model.step(self.state, action)

        return self.state, reward, self.done, dict()

    def render(self):
        pass

    def raw_state(self):
        return self.state

    def current_instance(self):
        return self.instance

    def max_num_actions(self):
        return len(self.state['remaining_ops'])

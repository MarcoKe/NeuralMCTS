from envs.gnn_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator
from envs.gnn_jsp_env.jsp_model import JobShopModel, init_adj_matrix, init_features
import gym
import numpy as np
from copy import deepcopy

# Parameters previously taken from the param_parser TODO take from arguments
init_quality_flag = False  # flag of whether init state quality is 0, True for 0
et_normalize_coef = 1000  # normalizing constant for feature LBs (end time), normalization way: fea/constant
rewardscale = 0.  # reward scale for positive rewards


class JobShopEnv(gym.Env):
    def __init__(self, instance_generator: JSPGenerator, **kwargs):
        self.jsp_generator = instance_generator
        self.model = JobShopModel()

        self.reset()

    def _generate_instance(self):
        self.instance = self.jsp_generator.generate()
        self.ops_per_job = self.instance.num_ops_per_job
        self.num_machines = self.instance.num_ops_per_job
        self.max_op_duration = self.instance.max_op_time
        self.num_jobs = self.instance.num_jobs
        self.num_ops = self.num_jobs * self.ops_per_job

        # possible operations to choose from next for each job (initialize with the first tasks for each job)
        possible_next_ops = np.arange(start=0, stop=self.num_ops, step=1).reshape(self.num_jobs, -1)[:, 0].astype(
            np.int64)
        # boolean values indicating whether all operations of a job have been scheduled or not
        mask = np.full(shape=self.num_jobs, fill_value=0, dtype=bool)  # TODO fix?
        # start times of operations on each machine
        machine_start_times = -1 * np.ones((self.ops_per_job, self.num_jobs), dtype=np.int32)
        # operation IDs on each machine
        machine_op_ids = -1 * np.ones((self.ops_per_job, self.num_jobs), dtype=np.int32)
        # time at which the last scheduled operation ends for each job
        last_job_ops = [-1 for _ in range(self.num_jobs)]
        # time at which the last scheduled operation ends on each machine
        last_machine_ops = [-1 for _ in range(self.num_machines)]
        # 2D array with the same shape as the instance jobs, containing operations' end times if they are already
        # scheduled and 0 otherwise
        end_times = np.zeros_like(machine_start_times)

        adj_matrix = init_adj_matrix(self.num_ops, self.num_jobs)
        features = init_features(self.instance.jobs)
        schedule = [[] for _ in range(self.num_machines)]
        remaining_ops = deepcopy(self.instance.jobs)

        return {'remaining_ops': remaining_ops, 'schedule': schedule, 'end_times': end_times,
                'last_job_ops': last_job_ops, 'last_mch_ops': last_machine_ops, 'adj_matrix': adj_matrix,
                'features': features, 'possible_next_ops': possible_next_ops, 'mask': mask, 'jobs': self.instance.jobs,
                'machine_start_times': machine_start_times, 'machine_op_ids': machine_op_ids}

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

        return self.state, reward, self.done, {}

    def render(self):
        pass

    def raw_state(self):
        return self.state

    def current_instance(self):
        return self.instance

    def max_num_actions(self):
        return len(self.state['remaining_ops'])

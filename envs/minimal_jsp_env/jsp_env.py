from envs.minimal_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator
from envs.minimal_jsp_env.jsp_model import JobShopModel
import gym
from copy import deepcopy

class JobShopEnv(gym.Env):
    def __init__(self, jsp_generator: JSPGenerator):
        self.jsp_generator = jsp_generator
        self.model = JobShopModel()

        self.reset()

    def _generate_instance(self):
        self.instance = self.jsp_generator.generate()
        self.ops_per_job = self.instance.num_ops_per_job
        self.num_machines = self.instance.num_ops_per_job
        self.max_op_duration = self.instance.max_op_time
        self.num_jobs = self.instance.num_jobs

        schedule = [[] for _ in range(self.num_machines)]
        last_job_ops = [-1 for _ in range(self.num_jobs)]
        return {'remaining_operations': deepcopy(self.instance.jobs), 'schedule': schedule, 'last_job_ops': last_job_ops}

    def reset(self):
        self.done = False
        self.state = self._generate_instance()

        return self.state

    def step(self, action):
        self.state, reward, self.done = self.model.step(self.state, action)

        return self.state, reward, self.done, {}

    def render(self):
        pass
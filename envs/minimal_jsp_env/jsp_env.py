from envs.minimal_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator
from envs.minimal_jsp_env.jsp_model import JobShopModel
import gym


class JobShopEnv(gym.Env):
    def __init__(self, jsp_generator: JSPGenerator):
        self.jsp_generator = jsp_generator
        self.model = JobShopModel()

        self.reset()

    def _generate_instance(self):
        instance = self.jsp_generator.generate()
        self.ops_per_job = instance.num_ops_per_job
        self.num_machines = instance.num_ops_per_job
        self.max_op_duration = instance.max_op_time
        self.num_jobs = instance.num_jobs

        schedule = [[] for i in range(self.num_machines)]
        return {'remaining_operations': instance.jobs, 'schedule': schedule}

    def reset(self):
        self.done = False
        self.state = self._generate_instance()
        return self.state

    def step(self, action):
        self.state, reward, self.done = self.model.step(self.state, action)
        return self.state, reward, self.done, {}

    def render(self):
        pass
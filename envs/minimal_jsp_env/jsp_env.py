from envs.minimal_jsp_env.util.visualization.gantt_visualizer import create_gantt
from envs.minimal_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator
from envs.minimal_jsp_env.jsp_model import JobShopModel
import gym
from copy import deepcopy

class JobShopEnv(gym.Env):
    def __init__(self, instance_generator: JSPGenerator, **kwargs):
        self.generator = instance_generator
        self.model = JobShopModel()

        self.reset()

    def set_instance(self, instance):
        self.done = False
        self.instance = instance
        self.ops_per_job = self.instance.num_ops_per_job
        self.num_machines = self.instance.num_ops_per_job
        self.max_op_duration = self.instance.max_op_time
        self.num_jobs = self.instance.num_jobs

        schedule = [[] for _ in range(self.num_machines)]
        last_job_ops = [-1 for _ in range(self.num_jobs)]

        s_ = {'remaining_operations': deepcopy(self.instance.jobs), 'schedule': schedule,
                'last_job_ops': last_job_ops}

        self.state = s_
        return self.state

    def _generate_instance(self):
        instance = self.generator.generate()
        self.set_instance(instance)

    def reset(self):
        self.done = False
        self._generate_instance()

        return self.state

    def set_state(self, state):
        self.state = state
        if len(state['remaining_operations']) > 0:
            self.done = False

    def step(self, action):
        self.state, reward, self.done = self.model.step(self.state, action)

        return self.state, reward, self.done, {}

    def render(self):
        create_gantt(self.state['schedule'])

    def raw_state(self):
        return self.state

    def current_instance(self):
        return self.instance
    
    def max_num_actions(self):
        return len(self.state['remaining_operations'])

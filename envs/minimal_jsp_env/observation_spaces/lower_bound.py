import gym
import numpy as np
import math

"""
Creates a vector containing the following information:
- whether each operation has been scheduled or not
- each operations lower bound makespan (a la learning to dispatch)
"""


class LowerBoundObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super(LowerBoundObservationSpace, self).__init__(env)
        vector_length = env.num_jobs * env.ops_per_job * 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (vector_length,), np.float32)

    def calculate_lower_bounds(self, schedule, remaining_ops):
        lower_bounds = np.zeros(self.env.num_jobs * self.env.ops_per_job)
        scheduled = np.zeros(self.env.num_jobs * self.env.ops_per_job)

        for machine_schedule in schedule:
            for entry in machine_schedule:
                op, start_time, end_time = entry
                lower_bounds[op.unique_op_id] = end_time
                scheduled[op.unique_op_id] = 1

        for job in remaining_ops:
            for op in job:
                if op.op_id == 0:
                    lower_bounds[op.unique_op_id] = op.duration
                else:
                    lower_bounds[op.unique_op_id] = lower_bounds[op.unique_op_id-1] + op.duration

        return lower_bounds, scheduled

    def norm_coef(self, max_duration, num_ops_per_job, num_jobs):
        i = 10
        while i < max_duration * num_ops_per_job * num_jobs:
            i *= 10

        return i

    def observation(self, observation):
        schedule = observation['schedule']
        remaining_ops = observation['remaining_operations']
        lower_bounds, scheduled = self.calculate_lower_bounds(schedule, remaining_ops)
        lower_bounds /= self.norm_coef(self.env.max_op_duration, self.env.ops_per_job, self.env.num_jobs)

        return np.concatenate((lower_bounds, scheduled))



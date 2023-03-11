import gym
import numpy as np


# Parameters previously taken from the param_parser TODO take from arguments
high = 9  # duration upper bound
low = 1  # duration lower bound

class NaiveObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super(NaiveObservationSpace, self).__init__(env)
        low_bounds = np.tile(np.array([low, 0], dtype=np.float32), (self.env.num_ops, 1))
        high_bounds = np.tile(np.array([high, 1], dtype=np.float32), (self.env.num_ops, 1))
        self.observation_space = gym.spaces.Dict(
            {"adj_matrix": gym.spaces.MultiBinary(n=self.env.state['adj_matrix'].shape),
             "features": gym.spaces.Box(low=low_bounds, high=high_bounds,
                                        shape=self.env.state['features'].shape, dtype=np.float32)})

    def observation(self, observation):
        """
        TODO
        :param observation: state returned by model.step(state, action)
        """
        # schedule = observation['schedule']
        # remaining_ops = observation['remaining_ops']
        #
        # machine_job_ids = [-1 for _ in range(self.env.num_machines)]
        # machine_durations = [-1 for _ in range(self.env.num_machines)]
        # for machine, machine_schedule in enumerate(schedule):
        #     if len(machine_schedule) > 0:
        #         last_op, start_time, end_time = machine_schedule[-1]
        #         machine_job_ids[machine] = self.normalize(last_op.job_id, 0, self.env.num_jobs)
        #         machine_durations[machine] = self.normalize(last_op.duration, 0, self.env.max_op_duration)
        # machine_obs = machine_job_ids + machine_durations
        #
        # rem_job_ids = []
        # rem_durations = []
        #
        # for job in remaining_ops:
        #     job_ids = [-1 for _ in range(self.env.ops_per_job)]
        #     durations = [-1 for _ in range(self.env.ops_per_job)]
        #     for op in job:
        #         job_ids[op.op_id] = self.normalize(op.job_id, 0, self.env.num_jobs)
        #         durations[op.op_id] = self.normalize(op.duration, 0, self.env.max_op_duration)
        #
        #     rem_job_ids += job_ids
        #     rem_durations += durations
        #
        # remaining_ops_obs = rem_job_ids + rem_durations
        #
        # obs = machine_obs + remaining_ops_obs
        # return np.array(obs).astype(np.float32)
        return {"adj_matrix": observation['adj_matrix'], "features": observation['features']}

    def normalize(self, val, min, max):
        return val - min / (max - min)

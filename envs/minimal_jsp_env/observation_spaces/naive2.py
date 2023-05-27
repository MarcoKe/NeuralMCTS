import gym
import numpy as np
import math

"""
Creates a vector containing the following information:
- the job id of the latest scheduled operation o_m on each machine m  (normalized)
- the relative duration of the latest scheduled operation on each machine. 
  relative with respect to min(finish_time(o_m) forall m) and max(finish_time(o_m) forall m)
  s.t. relative_duration_m = (finish_time(o_m) - min(...)) / (max(...) - min(...))
- on machines without ANY scheduled operations, both of the above are given as -1
  
- for every operation: its corresponding job id and its duration (normalized)
- for finished operations, these are represented as -1 
"""

class NaiveObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super(NaiveObservationSpace, self).__init__(env)
        vector_length = env.num_machines * 2  # machine state. 2 = (job_id, duration)
        vector_length += env.num_jobs * env.ops_per_job * 2  # remaining jobs state. 3 = (job_id, duration)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (vector_length,), np.float32)

    def observation(self, observation):
        schedule = observation['schedule']
        remaining_ops = observation['remaining_operations']

        machine_job_ids = [-1 for _ in range(self.env.num_machines)]
        machine_durations = [-1 for _ in range(self.env.num_machines)]
        earliest, latest = self.finish_time_extrema(schedule)
        for machine, machine_schedule in enumerate(schedule):
            if len(machine_schedule) > 0:
                last_op, start_time, end_time = machine_schedule[-1]
                machine_job_ids[machine] = self.normalize(last_op.job_id, 0, self.env.num_jobs)
                machine_durations[machine] = self.normalize(end_time-earliest, 0, latest-earliest)
        machine_obs = machine_job_ids + machine_durations

        rem_job_ids = []
        rem_durations = []

        for job in remaining_ops:
            job_ids = [-1 for _ in range(self.env.ops_per_job)]
            durations = [-1 for _ in range(self.env.ops_per_job)]
            for op in job:
                job_ids[op.op_id] = self.normalize(op.job_id, 0, self.env.num_jobs)
                durations[op.op_id] = self.normalize(op.duration, 0, self.env.max_op_duration)

            rem_job_ids += job_ids
            rem_durations += durations

        remaining_ops_obs = rem_job_ids + rem_durations

        obs = machine_obs + remaining_ops_obs
        return np.array(obs).astype(np.float32)

    def finish_time_extrema(self, schedule):
        """
        finds the earliest and latest finish time over all machines in a given schedule.
        i.e. among all the last operations scheduled on all machines, what is the earliest and latest finish time
        """
        earliest = math.inf
        latest = 0
        for machine, machine_schedule in enumerate(schedule):
            if len(machine_schedule) > 0:
                last_op, start_time, end_time = machine_schedule[-1]
                if end_time < earliest:
                    earliest = end_time

                if end_time > latest:
                    latest = end_time

        if earliest == math.inf:
            earliest = 0
        return earliest, latest

    def normalize(self, val, min, max):
        if min == max:
            return 0
        # print(val, min, max, (val - min) / (max - min))
        return (val - min) / (max - min)

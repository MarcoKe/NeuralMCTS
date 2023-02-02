import gym
import numpy as np


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
        for machine, machine_schedule in enumerate(schedule):
            if len(machine_schedule) > 0:
                last_op, start_time, end_time = machine_schedule[-1]
                machine_job_ids[machine] = self.normalize(last_op.job_id, 0, self.env.num_jobs)
                machine_durations[machine] = self.normalize(last_op.duration, 0, self.env.max_op_duration)
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

    def normalize(self, val, min, max):
        return val - min / (max - min)

    def normalize_observation(self, machines_remaining_processing_time, machines_queue_time, operations_dur_sums,
                              next_operation_dur, next_machine_type, current_time):
        machines_remaining_processing_time = [e / self.sim.max_op_time() for e in machines_remaining_processing_time]
        machines_queue_time = [e / (self.sim.num_jobs() * self.sim.max_op_time()) for e in machines_queue_time]
        operations_dur_sums = [e / (self.sim.max_op_time() * self.sim.max_ops_per_job()) for e in operations_dur_sums]
        next_operation_dur = [e / self.sim.max_op_time() for e in next_operation_dur]
        next_machine_type = [e / (self.sim.num_machines() - 1) for e in next_machine_type]
        current_time = current_time / (self.sim.num_jobs() * self.sim.max_ops_per_job() * self.sim.max_op_time())

        return machines_remaining_processing_time, machines_queue_time, operations_dur_sums, next_operation_dur, next_machine_type, current_time

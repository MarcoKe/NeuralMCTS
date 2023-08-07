import gym
import numpy as np
import random
from envs.minimal_jsp_env.observation_spaces.lower_bound import LowerBoundObservationSpace

"""
Creates a vector containing the following information:
- whether each operation has been scheduled or not
- each operations lower bound makespan (a la learning to dispatch)
"""


class LowerBoundWrapperObservationSpace(LowerBoundObservationSpace):
    def __init__(self, env):
        super(LowerBoundWrapperObservationSpace, self).__init__(env)
        print("using wrapped obs space")
        size = 15
        self.target_size = size
        vector_length = size**2 * 2
        assert vector_length > (env.num_jobs * env.ops_per_job * 2), "Can only wrap smaller observation into bigger observation"
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (vector_length,), np.float32)



    def observation(self, observation):
        schedule = observation['schedule']
        remaining_ops = observation['remaining_operations']
        lower_bounds, scheduled = self.calculate_lower_bounds(schedule, remaining_ops)
        lower_bounds /= self.norm_coef(self.env.max_op_duration, self.env.ops_per_job, self.env.num_jobs)

        ###
        jobs_bounds = np.split(lower_bounds, self.env.num_jobs)
        jobs_scheduled = np.split(scheduled, self.env.num_jobs)

        # list containing bound array for every job
        bounds_array_list = [np.zeros(self.target_size) for _ in range(self.target_size)]
        scheduled_array_list = [np.zeros(self.target_size) for _ in range(self.target_size)]
        list_indices = sorted(random.sample([i for i in range(self.target_size)], self.env.num_jobs))

        for job_b, job_s, list_index in zip(jobs_bounds, jobs_scheduled, list_indices):
            # init bounds for one empty job
            new_job_bounds = np.zeros(self.target_size)
            new_job_scheduled = np.zeros(self.target_size)


            # select indices in which to insert old data
            indices = sorted(random.sample([i for i in range(self.target_size)], self.env.ops_per_job))

            # insert old data into new empty bound array
            for data_b, data_s, index in zip(job_b, job_s, indices):
                new_job_bounds[index] = data_b
                new_job_scheduled[index] = data_s

            bounds_array_list[list_index] = new_job_bounds
            scheduled_array_list[list_index] = new_job_scheduled

        lower_bounds = np.concatenate(bounds_array_list)
        scheduled = np.concatenate(scheduled_array_list)


        ###

        return np.concatenate((lower_bounds, scheduled))


if __name__ == '__main__':
    lower_bounds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    scheduled = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

    num_jobs = 3
    ops_per_job = 3
    target_size = 6
    ###
    jobs_bounds = np.split(lower_bounds, num_jobs)
    jobs_scheduled = np.split(scheduled, num_jobs)

    # list containing bound array for every job
    bounds_array_list = [np.zeros(target_size) for _ in range(target_size)]
    scheduled_array_list = [np.zeros(target_size) for _ in range(target_size)]
    list_indices = sorted(random.sample([i for i in range(target_size)], num_jobs))

    for job_b, job_s, list_index in zip(jobs_bounds, jobs_scheduled, list_indices):
        # init bounds for one empty job
        new_job_bounds = np.zeros(target_size)
        new_job_scheduled = np.zeros(target_size)

        # select indices in which to insert old data
        indices = sorted(random.sample([i for i in range(target_size)], ops_per_job))

        # insert old data into new empty bound array
        for data_b, data_s, index in zip(job_b, job_s, indices):
            new_job_bounds[index] = data_b
            new_job_scheduled[index] = data_s

        bounds_array_list[list_index] = new_job_bounds
        scheduled_array_list[list_index] = new_job_scheduled

    lower_bounds = np.concatenate(bounds_array_list)
    scheduled = np.concatenate(scheduled_array_list)

    ###

    print(lower_bounds)
    print(scheduled)

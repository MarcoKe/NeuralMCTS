import random
import numpy as np
from envs.minimal_jsp_env.entities import Operation, JSPInstance
from envs.minimal_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator
from typing import List


class RandomJSPGenerator(JSPGenerator):
    def __init__(self, num_jobs: int, num_operations: int, max_op_duration: int = 9):
        self.num_jobs = num_jobs
        self.num_operations = num_operations
        self.max_op_duration = max_op_duration

    def generate(self):
        jobs = []
        for i in range(0, self.num_jobs):
            operations = []
            for j in range(0, self.num_operations):
                op_id = j
                duration = random.randint(1, self.max_op_duration)
                type = random.randint(0, self.num_operations - 1)

                operations.append(Operation(i, op_id, type, duration))
            jobs.append(operations)

        return JSPInstance(jobs, num_ops_per_job=self.num_operations, max_op_time=self.max_op_duration)


class RandomJSPGeneratorPool(JSPGenerator):
    def __init__(self, num_jobs: int, num_operations: int, max_op_duration: int = 9):
        self.num_jobs = num_jobs
        self.num_operations = num_operations
        self.max_op_duration = max_op_duration
        self.pool_size = self.num_jobs*self.num_operations
    
    def generate(self):
        operations_pool = [(random.randint(0, self.num_operations - 1), random.randint(1, self.max_op_duration)) for i in range(0, self.pool_size)] # (type, duration)

        jobs = []
        for job_id in range(0, self.num_jobs):

            index_list = np.random.choice(range(self.pool_size), self.num_operations)
            operation_parameters_list = [operations_pool[i] for i in index_list]
            operations = [Operation(job_id, op_id, i[0], i[1]) for op_id, i in enumerate(operation_parameters_list)]
            jobs.append(operations)

        return JSPInstance(jobs, num_ops_per_job=self.num_operations, max_op_time=self.max_op_duration)


class RandomJSPGeneratorOperationDistirbution(JSPGenerator):
    def __init__(self, num_jobs: int, num_operations: int, max_op_duration: int = 9):
        self.num_jobs = num_jobs
        self.num_operations = num_operations
        self.max_op_duration = max_op_duration
        self.pool_size = self.num_jobs*self.num_operations
    
    def generate(self, operation_distribution: List):
        assert len(operation_distribution) <= self.pool_size, "The size of the operation_distribution list does not match the pool_size."

        random_operations = [(random.randint(0, self.num_operations - 1), random.randint(1, self.max_op_duration)) for i in range(0, len(operation_distribution))] # generate len(operation_distribution) random operations (type, duration)
        
        operations_pool = []
        for distr, operation in zip(operation_distribution, random_operations):
            operations_pool += int(self.pool_size*distr)*[operation]
        random.shuffle(operations_pool)
        
        jobs = []
        for job_id in range(0, self.num_jobs):
            job_operations = operations_pool[self.num_operations*job_id:self.num_operations*(job_id+1)]
            operations = [Operation(job_id, op_id, type, duration) for op_id, (type, duration) in enumerate(job_operations)]
            jobs.append(operations)

        return JSPInstance(jobs, num_ops_per_job=self.num_operations, max_op_time=self.max_op_duration)


class RandomJSPGeneratorWithJobPool(JSPGenerator):
    def __init__(self, num_jobs: int, num_operations: int, max_op_duration: int = 9, job_pool_size: int = 10, operation_distribution: List = None):
        self.num_jobs = num_jobs
        self.num_operations = num_operations
        self.max_op_duration = max_op_duration
        self.job_pool_size = job_pool_size
        self.operation_distribution = operation_distribution

        if self.operation_distribution:
            pool_generator = RandomJSPGeneratorOperationDistirbution(
                num_jobs=self.job_pool_size, 
                num_operations=self.num_operations, 
                max_op_duration=self.max_op_duration)
            self.job_pool = pool_generator.generate(operation_distribution)
        else:
            pool_generator = RandomJSPGeneratorPool(
                num_jobs=self.job_pool_size,
                num_operations=self.num_operations, 
                max_op_duration=self.max_op_duration)

            self.job_pool = pool_generator.generate()

    def generate(self):
               
        index_list = np.random.choice(range(self.job_pool_size), self.num_jobs)
        jobs = [self.job_pool.jobs[i] for i in index_list]

        return JSPInstance(jobs, num_ops_per_job=self.num_operations, max_op_time=self.max_op_duration)

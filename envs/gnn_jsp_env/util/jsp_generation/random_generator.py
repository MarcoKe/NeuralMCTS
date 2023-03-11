import random
from envs.gnn_jsp_env.entities import Operation, JSPInstance
from envs.gnn_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator


class RandomJSPGenerator(JSPGenerator):
    def __init__(self, num_jobs: int, num_operations: int, max_op_duration: int = 9):
        self.num_jobs = num_jobs
        self.num_operations = num_operations
        self.max_op_duration = max_op_duration

    def generate(self):
        """
        Generates jobs consisting of operations with random durations and orders in which to be carried out,
        and returns a JSPInstance based on these jobs
        """
        jobs = []
        for i in range(0, self.num_jobs):
            operations = []
            for j in range(0, self.num_operations):
                op_id = j
                duration = random.randint(1, self.max_op_duration)
                machine_type = random.randint(0, self.num_operations - 1)

                operations.append(Operation(i, op_id, machine_type, duration))
            jobs.append(operations)

        return JSPInstance(jobs, num_ops_per_job=self.num_operations, max_op_time=self.max_op_duration)

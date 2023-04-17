from collections import namedtuple
from typing import List
from envs.minimal_jsp_env.util.jsp_solver import JSPSolver
from envs.minimal_jsp_env.util.jsp_generation.entropy_functions import calculate_entropy_from_operations_list, collect_all_operations

Operation = namedtuple("Operation", ["job_id", "op_id", "unique_op_id", "machine_type", "duration"])


class JSPInstance:
    def __init__(self, jobs: List, num_ops_per_job: int=None, num_machines: int=None, max_op_time: int=None,
                 id: str=None, opt_time: float=None, calculate_opt_time=True):
        self.jobs = jobs  # jobs consisting of operations
        self.num_jobs = len(jobs)
        self.num_ops_per_job = num_ops_per_job if num_ops_per_job else len(self.jobs[0])
        self.num_machines = num_machines
        self.max_op_time = max_op_time if max_op_time else max([op.duration for job in jobs for op in job])
        self.id = id

        if opt_time:
            self.opt_time = opt_time
        elif calculate_opt_time:
            self.opt_time = JSPSolver().solve_from_job_list(jobs)

        all_operations = collect_all_operations(jobs)
        total_entropy = calculate_entropy_from_operations_list(all_operations)
        self.relative_entropy = total_entropy / calculate_entropy_from_operations_list(list(range(len(all_operations))))

from collections import namedtuple
from typing import List
from envs.minimal_jsp_env.util.jsp_solver import JSPSolver

from scipy.stats import entropy
from collections import Counter

Operation = namedtuple("Operation", ["job_id", "op_id", "machine_type", "duration"])

def calculate_entropy_from_operations_list(operation_types: List) -> float:
    counts = Counter(operation_types)
    probabilities = [count/len(operation_types) for count in counts.values()]
    entropy_val = entropy(probabilities)
    return entropy_val

def collect_all_operations(jobs: List[List[Operation]]) -> List[tuple]:
    all_operations = []
    for job in jobs:
        job_operations = [(i.machine_type, i.duration) for i in job]
        all_operations += job_operations
    return all_operations


class JSPInstance:
    def __init__(self, jobs: List, num_ops_per_job: int=None, max_op_time: int=None,
                 id: str=None, opt_time: float=None):
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_ops_per_job = num_ops_per_job #todo infer if not given
        self.max_op_time = max_op_time #todo infer if not given
        self.id = id

        if opt_time:
            self.opt_time = opt_time
        else:
            self.opt_time = JSPSolver().solve(jobs)

        all_operations = collect_all_operations(jobs)
        total_entropy = calculate_entropy_from_operations_list(all_operations)
        self.relative_entropy = total_entropy / calculate_entropy_from_operations_list(list(range(len(all_operations))))


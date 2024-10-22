from collections import namedtuple
from typing import List
from envs.minimal_jsp_env.util.jsp_solver import JSPSolver
from envs.minimal_jsp_env.util.jsp_generation.entropy_functions import calculate_entropy_from_operations_list

# todo: make this prettier. unique_op_id is a unique op id across all jobs. op_id should probably be renamed "precedence"
Operation = namedtuple("Operation", ["job_id", "op_id", "unique_op_id", "machine_type", "duration"])

def collect_all_operations(jobs: List) -> List[tuple]:
    """Collect a list of all operations in the given jobs."""
    all_operations = []
    for job in jobs:
        job_operations = [(i.machine_type, i.duration) for i in job]
        all_operations += job_operations
    return all_operations


class JSPInstance:
    def __init__(self, jobs: List, num_ops_per_job: int=None, max_op_time: int=None, num_machines: int=None,
                 id: str=None, opt_time: float=None, spt_time: float=None, intra_instance_op_entropy=None):
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_ops_per_job = num_ops_per_job #todo infer if not given
        self.max_op_time = max_op_time #todo infer if not given
        self.num_machines = num_machines if num_machines else num_ops_per_job
        self.id = id
        self.spt_time = spt_time
        self.opt_time = opt_time
        self.intra_instance_op_entropy = intra_instance_op_entropy

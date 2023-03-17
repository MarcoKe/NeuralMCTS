from collections import namedtuple
from typing import List

Operation = namedtuple("Operation", ["job_id", "op_id", "machine_type", "duration"])


class JSPInstance:
    def __init__(self, jobs: List, num_ops_per_job: int=None, num_machines: int=None, max_op_time: int=None,
                 id: str=None, opt_time: float=None):
        self.jobs = jobs  # jobs consisting of operations
        self.num_jobs = len(jobs)
        self.num_ops_per_job = num_ops_per_job if num_ops_per_job else len(self.jobs[0])
        self.num_machines = num_machines
        self.max_op_time = max_op_time if max_op_time else max([op.duration for job in jobs for op in job])
        self.id = id
        self.opt_time = opt_time


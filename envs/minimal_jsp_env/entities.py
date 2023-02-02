from collections import namedtuple
from typing import List

Operation = namedtuple("Operation", ["job_id", "op_id", "machine_type", "duration"])


class JSPInstance:
    def __init__(self, jobs: List, num_ops_per_job: int=None, max_op_time: int=None,
                 id: str=None, opt_time: float=None):
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_ops_per_job = num_ops_per_job #todo infer if not given
        self.max_op_time = max_op_time #todo infer if not given
        self.id = id
        self.opt_time = opt_time


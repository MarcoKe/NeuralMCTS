import json
from envs.minimal_jsp_env.entities import JSPInstance, Operation


class JSONReader:
    @staticmethod
    def read_instance(path: str) -> JSPInstance:

        with open(path, 'r') as file:
            input_file = json.load(file)

        jobs = []

        for job_id, (job_machine_types, job_durations) in enumerate(zip(input_file['machine_types'], input_file['durations'])):
            operations = []
            for op_id, (operation_machine_type, operation_duration) in enumerate(zip(job_machine_types, job_durations)):
                unique_op_id = op_id + job_id * len(input_file['durations'])
                operations.append(Operation(job_id, op_id, unique_op_id, operation_machine_type, operation_duration))

            jobs.append(operations)

        return JSPInstance(jobs, num_ops_per_job=input_file['num_ops_per_job'], max_op_time=input_file['max_op_time'],
                    id=path.split('/')[-1], opt_time=input_file['opt_time'], spt_time=0)


from envs.gnn_jsp_env.entities import Operation, JSPInstance
import json
from typing import List


class JSPReaderJSON:
    def read_instance(self, path: str) -> JSPInstance:

        with open(path, 'r') as file:
            input_file = json.load(file)

        jobs = []

        for job_id, (job_machine_types, job_durations) in enumerate(
                zip(input_file['machine_types'], input_file['durations'])):

            operations = []
            for operation_id, (operation_machine_type, operation_duration) in enumerate(
                    zip(job_machine_types, job_durations)):
                operations.append(Operation(job_id, operation_id, operation_machine_type, operation_duration))

            jobs.append(operations)

        return JSPInstance(jobs, input_file['num_ops_per_job'], input_file['max_op_time'], input_file['id'],
                           input_file['opt_time'])


class JSPReaderGoogleOR:
    def read_instance(self, path: str) -> List:

        with open(path, 'r') as file:
            input_file = json.load(file)

        jobs = []

        for job_machine_types, job_durations in zip(input_file['machine_types'], input_file['durations']):

            operations = []
            for operation_machine_type, operation_duration in zip(job_machine_types, job_durations):
                operations.append([operation_machine_type, operation_duration])

            jobs.append(operations)

        return jobs

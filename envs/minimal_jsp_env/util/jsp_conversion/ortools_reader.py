import json
from typing import List


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

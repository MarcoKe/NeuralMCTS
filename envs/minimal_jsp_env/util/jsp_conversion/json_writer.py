from envs.minimal_jsp_env.entities import JSPInstance
import json


class JSPWriterJSON:
    @staticmethod
    def get_job_matrices(jsp_instance: JSPInstance, display: bool = True):
        machine_types = []
        durations = []
        for job in jsp_instance.jobs:
            machine_types.append([operation.machine_type for operation in job])
            durations.append([operation.duration for operation in job])
            if display:
                print([f"({operation.machine_type}, {operation.duration})" for operation in job])

        return machine_types, durations

    def write_instance(self, instance: JSPInstance, path: str, id: str):
        machine_types, durations = self.get_job_matrices(instance, display=False)
        json_file = {
            'machine_types': machine_types, 
            'durations': durations, 
            'num_ops_per_job' : instance.num_ops_per_job,
            'max_op_time': instance.max_op_time,
            'id': id,
            'opt_time': instance.opt_time
            }
        with open(path, 'w') as file:
            json.dump(json_file, file)
import json
from envs.minimal_jsp_env.util.jsp_conversion.converter import JSPReader
from envs.minimal_jsp_env.entities import Operation, JSPInstance

class SamsonovReader(JSPReader):
    def read_instance(self, path):
        with open(path, 'r') as input_file:
            input = json.load(input_file)

            return self._convert_to_internal_rep(input)

    @staticmethod
    def _convert_to_internal_rep(input):
        jobs_input = input['jssp_instance']

        job_list = []
        for job_id, (jd, jm) in enumerate(zip(jobs_input['durations'], jobs_input['machines'])):
            op_list = []
            for op_id, (d, m) in enumerate(zip(jd, jm)):
                unique_op_id = op_id + job_id * len(jobs_input['durations'])
                op_list.append(Operation(job_id, op_id, unique_op_id, m, d))

            job_list.append(op_list)

        op_entropy = None if not 'intra_instance_operation_entropy' in input else input['intra_instance_operation_entropy']
        return JSPInstance(job_list, num_ops_per_job=input['n_ops_per_job'], max_op_time=input['max_op_time'],
                           id=input['jssp_identification'], opt_time=input['optimal_time'], spt_time=input['spt_time'], intra_instance_op_entropy=op_entropy)



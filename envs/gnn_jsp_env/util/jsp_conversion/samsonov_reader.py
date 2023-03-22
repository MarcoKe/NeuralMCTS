import json
from envs.gnn_jsp_env.util.jsp_conversion.converter import JSPReader
from envs.gnn_jsp_env.entities import Operation, JSPInstance


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
                op_list.append(Operation(job_id, op_id, m, d))

            job_list.append(op_list)

        return JSPInstance(job_list, input['n_ops_per_job'], input['max_op_time'], input['jssp_identification'],
                    input['optimal_time'])


import os


class DatasetAnalyzer():
    def __init__(self, dataset_path: str, instance_reader) -> None:
        self.dataset_path = dataset_path
        self.instance_reader = instance_reader

    def analyze_dataset(self):

        entropies = {}
        results = []

        for entropy in os.listdir(self.dataset_path):
            entropy_data_path = f"{self.dataset_path}/{entropy}"

            for file_name in os.listdir(entropy_data_path):
                instance = self.instance_reader.read_instance(f"{entropy_data_path}/{file_name}")
                
                file_data = {
                    'entropy': entropy, 
                    'file_name': file_name,
                    'relative_entropy': instance.relative_entropy, 
                    'num_jobs': instance.num_jobs, 
                    'num_ops_per_job': instance.num_ops_per_job, 
                    'max_op_time': instance.max_op_time, 
                    'opt_time': instance.opt_time,
                    }
                
                results.append(file_data)
                
        self.mean_num_ops_per_job = sum(r['num_ops_per_job'] for r in results) / len(results)
        self.mean_num_jobs = sum(r['num_jobs'] for r in results) / len(results)
        self.mean_max_op_time = sum(r['max_op_time'] for r in results) / len(results)

        self.mean_relative_entropies = {}
        for r in results:
            if r['entropy'] in self.mean_relative_entropies:
                self.mean_relative_entropies[r['entropy']] += r['relative_entropy']
            else:
                self.mean_relative_entropies[r['entropy']] = r['relative_entropy']
        
        num_entropies = len(self.mean_relative_entropies)
        for k, v in self.mean_relative_entropies.items():
            self.mean_relative_entropies[k] = v / (len(results)/num_entropies)

        self.all_results = {"mean_num_ops_per_job": self.mean_num_ops_per_job,
                            "mean_num_jobs": self.mean_num_jobs,
                            "mean_max_op_time": self.mean_max_op_time,
                            "mean_relative_entropies": self.mean_relative_entropies}
from envs.minimal_jsp_env.jsp_conversion.reader import JSPReaderJSON
from envs.minimal_jsp_env.entities import JSPInstance
import os
import pandas as pd

class DatasetAnalyzer():
    def __init__(self, dataset_path: str, instance_reader, jsp_instance) -> None:
        self.dataset_path = dataset_path
        self.instance_reader = instance_reader
        self.jsp_instance = jsp_instance

    def analyze_dataset(self):

        entropies = {}
        results_df = pd.DataFrame(columns=['entropy', 'file_name', 'relative_entropy', 'num_jobs', 'num_ops_per_job', 'max_op_time', 'opt_time'])

        for entropy in os.listdir(self.dataset_path):
            entropy_data_path = f"{self.dataset_path}/{entropy}"

            for file_name in os.listdir(entropy_data_path):
                instance = self.instance_reader.read_instance(f"{entropy_data_path}/{file_name}")
                
                file_data = pd.DataFrame({
                    'entropy': entropy, 
                    'file_name': file_name,
                    'relative_entropy': instance.relative_entropy, 
                    'num_jobs': instance.num_jobs, 
                    'num_ops_per_job': instance.num_ops_per_job, 
                    'max_op_time': instance.max_op_time, 
                    'opt_time': instance.opt_time
                    }, index=[0])
                
                results_df = results_df.append(file_data, ignore_index=True)
                
        self.mean_num_ops_per_job = results_df['num_ops_per_job'].values.mean()
        self.mean_num_jobs = results_df['num_jobs'].values.mean()
        self.mean_max_op_time = results_df['max_op_time'].values.mean()
        self.mean_opt_time = results_df['opt_time'].values.mean()

        self.mean_relative_entropies = results_df.groupby('entropy').mean(numeric_only=True)['relative_entropy'].to_dict()
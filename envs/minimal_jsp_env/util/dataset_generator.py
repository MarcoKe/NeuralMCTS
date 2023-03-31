import sys
import os

sys.path.append(os.getcwd())

from envs.minimal_jsp_env.util.jsp_generation.random_generator import RandomJSPGenerator, RandomJSPGeneratorPool, RandomJSPGeneratorOperationDistirbution, RandomJSPGeneratorWithJobPool
from envs.minimal_jsp_env.entities import Operation, JSPInstance
import json
import argparse


from envs.minimal_jsp_env.util.jsp_conversion.readers import JSPReaderGoogleOR, JSPReaderJSON
from envs.minimal_jsp_env.util.jsp_conversion.writers import JSPWriterJSON

import yaml

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script for generating jsp datasets.")
    parser.add_argument("--config_path", help="Path to the config file with generation parameters.", default="")
        
    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        config_file=yaml.safe_load(stream)

    num_jobs = config_file['num_jobs']
    num_operations = config_file['num_operations']
    max_op_duration = config_file['max_op_duration']

    # for now with predefined entropy distribution, next merge I'll add the entropy optimizer to generate the entropy distribution
    entropy0_2 = [0.5, 0.5]
    entropy0_3 = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    entropy0_4 = [0.08333333333333333, 0.16666666666666666, 0.25, 0.08333333333333333, 0.4166666666666667]
    entropy0_5 = [0.3333333333333333, 0.16666666666666666, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.16666666666666666]
    entropy0_6 = [0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.16666666666666666, 0.08333333333333333, 0.16666666666666666, 0.16666666666666666, 0.08333333333333333]
    entropy0_7 = [0.05555555555555555, 0.1111111111111111, 0.027777777777777776, 0.05555555555555555, 0.05555555555555555, 0.16666666666666666, 0.16666666666666666, 0.05555555555555555, 0.027777777777777776, 0.027777777777777776, 0.08333333333333333, 0.027777777777777776, 0.05555555555555555, 0.05555555555555555, 0.027777777777777776]
    entropy0_8 = [0.16666666666666666, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.05555555555555555, 0.027777777777777776, 0.05555555555555555, 0.027777777777777776, 0.027777777777777776, 0.05555555555555555, 0.05555555555555555, 0.027777777777777776, 0.027777777777777776, 0.05555555555555555, 0.027777777777777776, 0.08333333333333333, 0.05555555555555555, 0.08333333333333333, 0.027777777777777776]

    for entropy_val in range(2, 9, 1):

        dir_path = f"{config_file['target_path']}/entropy0_{entropy_val}/"
        os.makedirs(dir_path, exist_ok=True)

        for i in range(config_file['num_simulations']):
            if config_file['generator_type'] == 'random':
                instance = RandomJSPGenerator(num_jobs, num_operations, max_op_duration).generate()
            elif config_file['generator_type'] == 'random_pool':
                instance = RandomJSPGeneratorPool(num_jobs, num_operations, max_op_duration).generate()
            elif config_file['generator_type'] == 'random_pool_entropy':
                instance = RandomJSPGeneratorOperationDistirbution(num_jobs, num_operations, max_op_duration).generate(eval(f"entropy0_{entropy_val}"))
            elif config_file['generator_type'] == 'random_job_pool':
                instance = RandomJSPGeneratorWithJobPool(num_jobs, num_operations, max_op_duration).generate()
            elif config_file['generator_type'] == 'random_job_pool_entropy':
                instance = RandomJSPGeneratorWithJobPool(num_jobs, num_operations, max_op_duration, operation_distribution=eval(f"entropy0_{entropy_val}")).generate()
            
            file_name = f"entropy0_{entropy_val}_{i}.json"
            JSPWriterJSON().write_instance(instance, f"{dir_path}/{file_name}", file_name)
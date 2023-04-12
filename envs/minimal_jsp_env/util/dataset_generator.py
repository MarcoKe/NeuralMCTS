import sys
import os

sys.path.append(os.getcwd())

from envs.minimal_jsp_env.util.jsp_generation.random_generator import RandomJSPGenerator, RandomJSPGeneratorPool, RandomJSPGeneratorOperationDistirbution, RandomJSPGeneratorWithJobPool
from envs.minimal_jsp_env.entities import Operation, JSPInstance
from envs.minimal_jsp_env.util.jsp_generation.entropy_functions import EntropyOptimizer, calculate_entropy_from_operations_list
from scipy.stats import entropy
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
    optimizer_config = config_file['optimizer_config']

    optimizer = EntropyOptimizer(
        output_size=optimizer_config['output_size'], 
        hidden_size=optimizer_config['hidden_size'], 
        learning_rate=optimizer_config['learning_rate'], 
        num_epochs=optimizer_config['num_epochs'], 
        max_episodes=optimizer_config['max_episodes'], 
        precision=optimizer_config['precision'],
        )
    entropies = optimizer.find_entropies()

    max_entropy = calculate_entropy_from_operations_list(range(optimizer_config['output_size']))
    for ratio in entropies:
        output_entropy = entropy(entropies[ratio])
        relative_entropy = output_entropy/max_entropy
        entropy_diff = abs(relative_entropy-ratio)
        assert entropy_diff < 0.05, "One of the entropies does not meet the required ratio. Rerun the code again."



    for entropy_key in entropies.keys():

        entropy_val = str(entropy_key).replace(".", "_")

        dir_path = f"{config_file['target_path']}/entropy{entropy_val}/"
        os.makedirs(dir_path, exist_ok=True)

        for i in range(config_file['num_simulations']):
            if config_file['generator_type'] == 'random':
                instance = RandomJSPGenerator(num_jobs, num_operations, max_op_duration).generate()
            elif config_file['generator_type'] == 'random_pool':
                instance = RandomJSPGeneratorPool(num_jobs, num_operations, max_op_duration).generate()
            elif config_file['generator_type'] == 'random_pool_entropy':
                instance = RandomJSPGeneratorOperationDistirbution(num_jobs, num_operations, max_op_duration).generate(entropies[entropy_key])
            elif config_file['generator_type'] == 'random_job_pool':
                instance = RandomJSPGeneratorWithJobPool(num_jobs, num_operations, max_op_duration).generate()
            elif config_file['generator_type'] == 'random_job_pool_entropy':
                instance = RandomJSPGeneratorWithJobPool(num_jobs, num_operations, max_op_duration, operation_distribution=entropies[entropy_key]).generate()
            
            file_name = f"entropy{entropy_val}_{i}.json"
            JSPWriterJSON().write_instance(instance, f"{dir_path}/{file_name}", file_name)
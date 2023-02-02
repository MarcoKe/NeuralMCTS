import sys
from experiment_management.setup_experiment import setup_experiment

if __name__ == '__main__':
    exp_name = sys.argv[1]
    setup_experiment(exp_name)
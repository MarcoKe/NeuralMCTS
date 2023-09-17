import sys
from experiment_management.setup_experiment import setup_experiment, setup_budget_sensitivity_experiment, setup_env_test_experiment

if __name__ == '__main__':
    exp_name = sys.argv[1]

    if 'budget_sensitivity' in exp_name:
        setup_budget_sensitivity_experiment(exp_name)
    elif 'envs_test' in exp_name:
        setup_env_test_experiment(exp_name)
    else:
        setup_experiment(exp_name)
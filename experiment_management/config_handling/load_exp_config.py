from pathlib import Path
import yaml


def load_yml(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def load_exp_config(exp_name):
    general_config = load_yml(Path('data/config/config.yml'))

    experiments_path = Path(general_config['input']['experiment_configs'])
    agents_path = Path(general_config['input']['agent_configs'])
    environments_path = Path(general_config['input']['environment_configs'])

    # exp = load_yml((experiments_path / Path(exp_name + '.yml')))
    # agent_config = load_yml((agents_path / Path(exp['agent'] + '.yml')))
    # env_config = load_yml((environments_path / Path(exp['environment'] + '.yml')))
    exp = "exp"
    agent_config = "agent"
    env_config = "env"

    return general_config, exp_name, exp, agent_config, env_config

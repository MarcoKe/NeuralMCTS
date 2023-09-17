from pathlib import Path
import yaml


def load_yml(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def load_gen_config():
    general_config = load_yml(Path('data/config/config.yml'))

    experiments_path = Path(general_config['input']['experiment_configs'])
    agents_path = Path(general_config['input']['agent_configs'])
    environments_path = Path(general_config['input']['environment_configs'])

    return general_config, experiments_path, agents_path, environments_path


def load_exp_config(exp_name):
    general_config, experiments_path, agents_path, environments_path = load_gen_config()

    exp = load_yml((experiments_path / Path(exp_name + '.yml')))
    agent_config = load_yml((agents_path / Path(exp['agent'] + '.yml')))
    env_config = load_yml((environments_path / Path(exp['env'] + '.yml')))

    return general_config, exp_name, exp, agent_config, env_config


def load_sensitivity_exp_config(exp_name):
    general_config, experiments_path, agents_path, environments_path = load_gen_config()

    exp = load_yml((experiments_path / Path(exp_name + '.yml')))

    _, _, original_exp, agent_config, env_config = load_exp_config(exp['original_exp'])
    saved_agent_path = general_config['output']['saved_agents'] + '/' + exp['original_exp']
    agent_config['learned_policy']['location'] = saved_agent_path

    return general_config, exp_name, exp, original_exp, agent_config, env_config


def load_envs_test_exp_config(exp_name):
    general_config, experiments_path, agents_path, environments_path = load_gen_config()

    exp = load_yml((experiments_path / Path(exp_name + '.yml')))

    _, _, original_exp, agent_config, env_config = load_exp_config(exp['original_exp'])
    saved_agent_path = general_config['output']['saved_agents'] + '/' + exp['original_exp']
    saved_agent_path = saved_agent_path.split('/')
    saved_agent_path[-1] = 'final_' + saved_agent_path[-1]
    saved_agent_path = '/'.join(saved_agent_path)
    agent_config['learned_policy']['location'] = saved_agent_path

    eval_envs = []
    for env in exp['envs']:
        eval_envs.append(load_yml((environments_path / Path(env + '.yml'))))

    return general_config, exp_name, exp, original_exp, agent_config, eval_envs

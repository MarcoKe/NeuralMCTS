import uuid

import wandb
import numpy as np

from experiment_management.config_handling.load_exp_config import load_exp_config, load_sensitivity_exp_config, load_envs_test_exp_config
from sb3_contrib.common.wrappers import ActionMasker
from training.pimcts_trainer import MCTSPolicyImprovementTrainer
from training.stb3_trainer import Stb3Trainer
from evaluation.budget_evaluator import MCTSBudgetEvaluator
from experiment_management.utils import init_wandb, create_agent, create_env, create_model_free_agent, solver_factories
from evaluation.env_evaluator import EnvEvaluator


def setup_experiment(exp_name):
    general_config, exp_name, exp_config, agent_config, env_config = load_exp_config(exp_name)
    wandb_run = init_wandb(general_config, exp_name, exp_config, agent_config, env_config)

    env, eval_env, model = create_env(env_config)

    mcts_agent, model_free_agent = create_agent(general_config, env, model, agent_config)

    solver_factory = solver_factories.get(env_config['name'])
    solver = solver_factory.get('opt')  # todo
    trainer = MCTSPolicyImprovementTrainer(exp_config['name'], env, eval_env, mcts_agent, model_free_agent, wandb_run=wandb_run, solver=solver,
                                           **agent_config['training'], policy_improvement_iterations=exp_config['policy_improvement_iterations'])

    trainer.train()
    wandb_run.finish()


def setup_budget_sensitivity_experiment(exp_name):
    general_config, exp_name, exp_config, original_exp, agent_config, env_config = load_sensitivity_exp_config(exp_name)
    if 'saved_agent_path' in exp_config:
        agent_config['learned_policy']['location'] = exp_config['saved_agent_path']

    general_config['wandb']['project'] = 'neural_mcts_budget'
    wandb_run = init_wandb(general_config, exp_name, exp_config, agent_config, env_config)

    _, eval_env, model = create_env(env_config)
    mcts_agent, model_free_agent = create_agent(general_config, eval_env, model, agent_config)

    evaluator = MCTSBudgetEvaluator(exp_config['name'], eval_env, mcts_agent, model_free_agent, exp_config['budgets'], wandb_run)

    evaluator.evaluate()
    wandb_run.finish()


def setup_env_test_experiment(exp_name):
    general_config, exp_name, exp_config, agent_config, env_configs, model_free = load_envs_test_exp_config(exp_name)
    general_config['wandb']['project'] = 'neural_mcts_testset'

    evaluator = EnvEvaluator(general_config, exp_name, exp_config, agent_config, env_configs, model_free)
    evaluator.evaluate()


def setup_model_free_experiment(exp_name):
    general_config, exp_name, exp_config, agent_config, env_config = load_exp_config(exp_name)
    exp_name = str(uuid.uuid4())[:4] + '_' + exp_name
    wandb.tensorboard.patch(root_logdir=general_config['output']['tensorboard_logs'] + '/' + exp_name + '/PPO_0')
    wandb_run = init_wandb(general_config, exp_name, exp_config, agent_config, env_config)

    env, eval_env, _ = create_env(env_config)

    def mask_fn(env) -> np.ndarray:
        mask = np.array([False for _ in range(env.max_num_actions())])
        mask[env.model.legal_actions(env.raw_state())] = True
        return mask

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    eval_env = ActionMasker(eval_env, mask_fn)  # Wrap to enable masking

    agent = create_model_free_agent(general_config, env, agent_config, exp_name)

    trainer = Stb3Trainer(exp_config['name'], env, eval_env, agent, wandb_run, exp_config['training_steps'])
    trainer.train()
    trainer.evaluate()

    wandb_run.finish()


if __name__ == '__main__':
    # import glob
    #
    # configs = [
    #     # {'size': '6', 'type': 'inter_inst_job'},
    #     #        {'size': '10', 'type': 'inter_inst_op'},
    #     #        {'size': '10', 'type': 'inter_inst_job'},
    #     #        {'size': '15', 'type': 'intra_inst_op'},
    #            {'size': '15', 'type': 'inter_inst_op'},
    #            {'size': '15', 'type': 'inter_inst_job'}]
    #
    # for conf in configs:
    #     exp_id = 'envs_test_mf_' + conf['size'] + 'x' + conf['size'] + '_' + conf['type']
    #
    #     for exp in glob.glob('data/config/experiments/' + exp_id + '/*'):
    #         print(exp)
    #         exp = '/'.join(exp.split('/')[-2:]).split('.yml')[0]
    #         setup_env_test_experiment(exp)

    setup_model_free_experiment('model_free/15x15_intra_inst_op_08')

    # for exp in glob.glob('data/config/experiments/envs_test_intra_op_entropy_mf_10x10/*.yml'):
    #     exp = '/'.join(exp.split('/')[-2:]).split('.yml')[0]
    #     print(exp)
    #     setup_env_test_experiment(exp)
    # setup_budget_sensitivity_experiment('budget_sensitivity/budget_sensitivity_test')
    # setup_experiment("jsp_10x10")
    # setup_env_test_experiment('envs_test_entropy/jsp_jsp_neural_value_eval_full_expansion_puct_78a44fc6_3c5b1c15')
    # setup_experiment("jsp_test")
    # setup_model_free_experiment("model_free/jsp_test")
    # exps = ['model_free/inter_instance_op_02', 'model_free/inter_instance_op_03', 'model_free/inter_instance_op_04',
    #         'model_free/inter_instance_op_05', 'model_free/inter_instance_op_06', 'model_free/inter_instance_op_07',
    #         'model_free/inter_instance_op_08']
    #
    # for exp in exps:
    #     print(exp)
    #     setup_model_free_experiment(exp)
    # setup_experiment("naive2_ff_05/jsp_uct_neural_expansion_neural_rollout_eval_value_initialization_initialize_tree")

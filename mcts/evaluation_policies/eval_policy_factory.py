from util.object_factory import ObjectFactory
from mcts.evaluation_policies.neural_rollout_policy import NeuralRolloutPolicy
from mcts.evaluation_policies.neural_value_eval import NeuralValueEvalPolicy
from mcts.evaluation_policies.evaluation_policy import RandomRolloutPolicy
from mcts.evaluation_policies.mixed_policy import MixedPolicy

class EvalPolicyFactory(ObjectFactory):
    def get(self, key, **kwargs):
        if key == 'mixed':
            kwargs['policies'] = [self.create(p) for p in kwargs['policies']]
            # todo: the above only works for individual policies that do not accept parameters upon init

        return self.create(key, **kwargs)


eval_policy_factory = EvalPolicyFactory()
eval_policy_factory.register_builder('neural_rollout_eval', NeuralRolloutPolicy)
eval_policy_factory.register_builder('neural_value_eval', NeuralValueEvalPolicy)
eval_policy_factory.register_builder('random', RandomRolloutPolicy)
eval_policy_factory.register_builder('mixed', MixedPolicy)
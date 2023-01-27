from util.object_factory import ObjectFactory
from mcts.evaluation_policies.neural_rollout_policy import NeuralRolloutPolicy
from mcts.evaluation_policies.neural_value_eval import NeuralValueEvalPolicy


class EvalPolicyFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


eval_policies = EvalPolicyFactory()
eval_policies.register_builder('neural_rollout_eval', NeuralRolloutPolicy)
eval_policies.register_builder('neural_value_eval', NeuralValueEvalPolicy)

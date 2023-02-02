from util.object_factory import ObjectFactory
from mcts.expansion_policies.expansion_policy import ExpansionPolicy


class ExpansionPolicyFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


expansion_policy_factory = ExpansionPolicyFactory()
expansion_policy_factory.register_builder('full_expansion', ExpansionPolicy)

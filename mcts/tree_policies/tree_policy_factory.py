from util.object_factory import ObjectFactory
from mcts.tree_policies.exploitation_terms.avg_node_value import AvgNodeValueTerm
from mcts.tree_policies.exploitation_terms.max_node_value import MaxNodeValueTerm
from mcts.tree_policies.exploration_terms.puct_term import PUCTTerm
from mcts.tree_policies.exploration_terms.uct_term import UCTTerm
from mcts.tree_policies.tree_policy import RandomTreePolicy, UCTPolicy


class ExploitationTermFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


exploitation_factory = ExploitationTermFactory()
exploitation_factory.register_builder('avg_node_value', AvgNodeValueTerm)
exploitation_factory.register_builder('max_node_value', MaxNodeValueTerm)


class ExplorationTermFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


exploration_factory = ExplorationTermFactory
exploration_factory.register_builder('puct', PUCTTerm)
exploration_factory.register_builder('uct', UCTTerm)


class TreePolicyFactory(ObjectFactory):
    def get(self, key, **kwargs):
        if 'exploitation_term' in kwargs:
            kwargs['exploitation_term'] = exploitation_factory.get(kwargs['exploitation_term']['name'],
                                                                   **kwargs['exploitation_term']['params'])
            kwargs['exploration_term'] = exploration_factory.get(kwargs['exploration_term']['name'],
                                                                 **kwargs['exploration_term']['params'])
        return self.create(key, **kwargs)


tree_policy_factory = TreePolicyFactory()
tree_policy_factory.register_builder('random', RandomTreePolicy)
tree_policy_factory.register_builder('uct', UCTPolicy)

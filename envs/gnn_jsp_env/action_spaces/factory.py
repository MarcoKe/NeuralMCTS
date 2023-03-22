from util.object_factory import ObjectFactory
from envs.gnn_jsp_env.action_spaces.gnn import GNNActionSpace


class ActionSpaceFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


gnn_action_spaces = ActionSpaceFactory()
gnn_action_spaces.register_builder('naive', GNNActionSpace)

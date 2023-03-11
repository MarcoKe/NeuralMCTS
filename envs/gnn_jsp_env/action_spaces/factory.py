from util.object_factory import ObjectFactory
from envs.gnn_jsp_env.action_spaces.naive import NaiveActionSpace


class ActionSpaceFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


action_spaces = ActionSpaceFactory()
action_spaces.register_builder('naive', NaiveActionSpace)

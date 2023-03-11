from util.object_factory import ObjectFactory
from envs.gnn_jsp_env.observation_spaces.naive import NaiveObservationSpace

class ObservationSpaceFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


gnn_observation_spaces = ObservationSpaceFactory()
gnn_observation_spaces.register_builder('naive', NaiveObservationSpace)

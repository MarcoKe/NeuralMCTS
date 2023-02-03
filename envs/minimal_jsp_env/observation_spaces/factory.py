from util.object_factory import ObjectFactory
from envs.minimal_jsp_env.observation_spaces.naive import NaiveObservationSpace

class ObservationSpaceFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


observation_spaces = ObservationSpaceFactory()
observation_spaces.register_builder('naive', NaiveObservationSpace)

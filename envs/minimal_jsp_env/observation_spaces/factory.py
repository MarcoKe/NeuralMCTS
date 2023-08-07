from util.object_factory import ObjectFactory
from envs.minimal_jsp_env.observation_spaces.naive import NaiveObservationSpace
from envs.minimal_jsp_env.observation_spaces.naive2 import NaiveObservationSpace as NaiveObservationSpace2
from envs.minimal_jsp_env.observation_spaces.lower_bound import LowerBoundObservationSpace
from envs.minimal_jsp_env.observation_spaces.lower_bound_bigger_instance_wrapper import LowerBoundWrapperObservationSpace

class ObservationSpaceFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


observation_spaces = ObservationSpaceFactory()
observation_spaces.register_builder('naive', NaiveObservationSpace)
observation_spaces.register_builder('naive2', NaiveObservationSpace2)
observation_spaces.register_builder('lower_bound', LowerBoundObservationSpace)
observation_spaces.register_builder('lower_bound_wrapped', LowerBoundWrapperObservationSpace)



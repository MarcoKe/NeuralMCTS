from util.object_factory import ObjectFactory, ClassFactory
from envs.tsp.TSP import TSPGym, TSP
from envs.minimal_jsp_env.jsp_env import JobShopEnv
from envs.minimal_jsp_env.jsp_model import JobShopModel
from envs.minimal_jsp_env.observation_spaces.factory import observation_spaces
from envs.minimal_jsp_env.action_spaces.factory import action_spaces
from envs.minimal_jsp_env.reward_functions.factory import reward_functions
from envs.tsp.observation_spaces.factory import observation_spaces as tsp_observation_spaces
from envs.tsp.action_spaces.factory import action_spaces as tsp_action_spaces
from envs.tsp.reward_functions.factory import reward_functions as tsp_reward_functions
from envs.minimal_jsp_env.util.jsp_generation.instance_generator_factory import jsp_generators
from envs.minimal_jsp_env.util.jsp_solver import jsp_solvers

class EnvironmentFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


class ModelFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


class ObservationFactoryFactory(ClassFactory):
    def get(self, key, **kwargs):
        return self.get_class(key, **kwargs)


class ActionFactoryFactory(ClassFactory):
    def get(self, key, **kwargs):
        return self.get_class(key, **kwargs)


class RewardFactoryFactory(ClassFactory):
    def get(self, key, **kwargs):
        return self.get_class(key, **kwargs)


class InstanceFactoryFactory(ClassFactory):
    def get(self, key, **kwargs):
        return self.get_class(key, **kwargs)


class SolverFactoryFactory(ClassFactory):
    def get(self, key, **kwargs):
        return self.get_class(key, **kwargs)


env_factory = EnvironmentFactory()
env_factory.register_builder('tsp', TSPGym)
env_factory.register_builder('jsp_minimal', JobShopEnv)


model_factory = ModelFactory()
model_factory.register_builder('tsp', TSP)
model_factory.register_builder('jsp_minimal', JobShopModel)

observation_factories = ObservationFactoryFactory()
observation_factories.register_class('jsp_minimal', observation_spaces)
observation_factories.register_class('tsp', tsp_observation_spaces)


action_factories = ActionFactoryFactory()
action_factories.register_class('jsp_minimal', action_spaces)
action_factories.register_class('tsp', tsp_action_spaces)


reward_factories = RewardFactoryFactory()
reward_factories.register_class('jsp_minimal', reward_functions)
reward_factories.register_class('tsp', tsp_reward_functions)

instance_factories = InstanceFactoryFactory()
instance_factories.register_class('jsp_minimal', jsp_generators)
instance_factories.register_class('tsp', None) #todo

solver_factories = SolverFactoryFactory()
solver_factories.register_class('jsp_minimal', jsp_solvers)
solver_factories.register_class('tsp', None) #todo

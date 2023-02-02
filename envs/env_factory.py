from util.object_factory import ObjectFactory
from envs.TSP import TSPGym, TSP


class EnvironmentFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


class ModelFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


env_factory = EnvironmentFactory()
env_factory.register_builder('tsp', TSPGym)

model_factory = ModelFactory()
model_factory.register_builder('tsp', TSP)
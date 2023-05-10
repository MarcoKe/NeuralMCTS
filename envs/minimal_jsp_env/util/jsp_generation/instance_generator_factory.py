from envs.minimal_jsp_env.util.jsp_generation.random_generator import RandomJSPGenerator
from envs.minimal_jsp_env.util.jsp_generation.single_instance_generator import SingleInstanceGenerator
from envs.minimal_jsp_env.util.jsp_generation.multiple_instance_generator import MultipleInstanceGenerator, DeterministicMultiInstanceGenerator
from util.object_factory import ObjectFactory


class InstanceGeneratorFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


jsp_generators = InstanceGeneratorFactory()
jsp_generators.register_builder('random', RandomJSPGenerator)
jsp_generators.register_builder('single', SingleInstanceGenerator)
jsp_generators.register_builder('multiple', MultipleInstanceGenerator)
jsp_generators.register_builder('deterministic_multiple', DeterministicMultiInstanceGenerator)




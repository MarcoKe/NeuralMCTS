from envs.gnn_jsp_env.util.jsp_generation.random_generator import RandomJSPGenerator
from envs.gnn_jsp_env.util.jsp_generation.single_instance_generator import SingleInstanceGenerator
from envs.gnn_jsp_env.util.jsp_generation.multiple_instance_generator import MultipleInstanceGenerator
from util.object_factory import ObjectFactory


class InstanceGeneratorFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


gnn_generators = InstanceGeneratorFactory()
gnn_generators.register_builder('random', RandomJSPGenerator)
gnn_generators.register_builder('single', SingleInstanceGenerator)
gnn_generators.register_builder('multiple', MultipleInstanceGenerator)



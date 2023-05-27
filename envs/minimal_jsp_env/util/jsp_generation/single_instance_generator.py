import copy

from envs.minimal_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator
from envs.minimal_jsp_env.util.jsp_generation.random_generator import RandomJSPGenerator
from envs.minimal_jsp_env.entities import JSPInstance
from envs.minimal_jsp_env.util.jsp_conversion.converter import JSPReader
from envs.minimal_jsp_env.util.jsp_conversion.instance_reader_factory import instance_readers

class SingleInstanceRandomGenerator(JSPGenerator):
    def __init__(self, num_jobs: int = 6, num_operations: int = 6):
        self.num_jobs = num_jobs
        self.num_operations = num_operations

        random_generator = RandomJSPGenerator(self.num_jobs, self.num_operations, self.num_operations)
        self.instance = random_generator.generate()

    def generate(self) -> JSPInstance:
        return self.instance


class SingleInstanceGenerator(JSPGenerator):
    def __init__(self, path: str, reader: JSPReader):
        self.instance = reader.read_instance(path)

    def __init__(self, path: str, format: str, **kwargs):
        self.set_instance(path, format)

    def generate(self) -> JSPInstance:
        return copy.deepcopy(self.instance) #todo solve this generally

    def set_instance(self, path: str, format: str):
        reader = instance_readers.get(format)
        self.instance = reader.read_instance(path)

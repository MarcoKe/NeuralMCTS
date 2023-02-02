import glob
import random

from envs.minimal_jsp_env.util.jsp_generation.jsp_generator import JSPGenerator
from envs.minimal_jsp_env.entities import JSPInstance
from envs.minimal_jsp_env.util.jsp_conversion.converter import JSPReader
from envs.minimal_jsp_env.util.jsp_conversion.instance_reader_factory import instance_readers


class MultipleInstanceGenerator(JSPGenerator):
#     def __init__(self, path: str, reader: JSPReader):
#         self.instance = reader.read_instance(path)

    def __init__(self, path: str, format: str, **kwargs):
        self.reader = instance_readers.get(format)
        self.instances = [jsp_file for jsp_file in glob.glob(path + '*')]

    def generate(self) -> JSPInstance:
        random_instance = random.choice(self.instances)
        return self.reader.read_instance(random_instance)


class RepeatingMultipleInstanceGenerator(MultipleInstanceGenerator):
    def __init__(self, path: str, format: str, repetitions: int):
        super().__init__(path, format)
        print(len(self.instances))
        self.repetitions = repetitions
        self.count = 0
        self.current_instance = None

    def generate(self) -> JSPInstance:
        if self.count % self.repetitions == 0:
            self.current_instance = super().generate()

        self.count += 1
        return self.current_instance




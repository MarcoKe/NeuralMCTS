from envs.gnn_jsp_env.entities import JSPInstance


class JSPReader:
    def read_instance(self, path) -> JSPInstance:
        raise NotImplementedError


class JSPWriter:
    def write_instance(self, instance, path):
        raise NotImplementedError


class JSPConverter:
    def __init__(self, reader: JSPReader, writer: JSPWriter):
        self.reader = reader
        self.writer = writer

    def convert(self, jsp_instance_path, target_path):
        instance = self.reader.read_instance(jsp_instance_path)
        self.writer.write_instance(instance, target_path)

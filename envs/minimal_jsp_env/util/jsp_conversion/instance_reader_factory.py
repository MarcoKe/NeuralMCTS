from util.object_factory import ObjectFactory
from envs.minimal_jsp_env.util.jsp_conversion.samsonov_reader import SamsonovReader
from envs.minimal_jsp_env.util.jsp_conversion.json_reader import JSONReader


class ReaderFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


instance_readers = ReaderFactory()
instance_readers.register_builder('samsonov', SamsonovReader)
instance_readers.register_builder('json', JSONReader)

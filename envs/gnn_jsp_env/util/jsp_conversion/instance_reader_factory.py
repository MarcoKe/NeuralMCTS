from util.object_factory import ObjectFactory
from envs.gnn_jsp_env.util.jsp_conversion.samsonov_reader import SamsonovReader


class ReaderFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


instance_readers = ReaderFactory()
instance_readers.register_builder('samsonov', SamsonovReader)

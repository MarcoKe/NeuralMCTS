class ObjectFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


class ClassFactory:
    def __init__(self):
        self._classes = {}

    def register_class(self, key, class_):
        self._classes[key] = class_

    def get_class(self, key, **kwargs):
        class_ = self._classes.get(key)
        if not class_:
            raise ValueError(key)
        return class_


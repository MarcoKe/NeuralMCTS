class ExplorationTerm:
    def val(self, node):
        raise NotImplementedError

    def init_prior(self, node, **kwargs):
        pass

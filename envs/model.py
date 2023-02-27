class Model:
    @staticmethod
    def step(state, action):
        raise NotImplementedError

    @staticmethod
    def legal_actions(state):
        raise NotImplementedError


class ModelStepCounter(Model):
    def __init__(self, model):
        self.model = model
        self.count = 0

    def step(self, state, action):
        self.count += 1
        return self.model.step(state, action)

    def legal_actions(self, state):
        return self.model.legal_actions(state)

    def __getattr__(self, name):
        return getattr(self.model, name)



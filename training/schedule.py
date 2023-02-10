class Schedule:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps

    def value(self, step):
        raise NotImplementedError


class LinearSchedule(Schedule):
    def value(self, step):
        return step * (self.end - self.start) / self.steps
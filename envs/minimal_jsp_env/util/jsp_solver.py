from util.solver import Solver
from util.object_factory import ObjectFactory


class JSPSolverFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


class JSPSolver(Solver):
    @staticmethod
    def solve(instance):
        if instance.opt_time:
            return instance.opt_time
        else:
            return 0 # todo


jsp_solvers = JSPSolverFactory()
jsp_solvers.register_builder('opt', JSPSolver)


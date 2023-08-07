from util.solver import Solver
from util.object_factory import ObjectFactory


class JSPSolverFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


class JSPSolver(Solver):
    def solve(self, instance):
        solutions = dict()
        if instance.opt_time:
            solutions['opt'] = instance.opt_time
        else:
            return 0

        if instance.spt_time:
            solutions['spt'] = instance.spt_time

        return solutions


class SPTSolver(Solver):
    def solve(self, instance):
        pass


jsp_solvers = JSPSolverFactory()
jsp_solvers.register_builder('opt', JSPSolver)


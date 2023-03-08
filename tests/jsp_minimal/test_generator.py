from envs.minimal_jsp_env.util.jsp_generation.random_generator import RandomJSPGeneratorOperationDistirbution, RandomJSPGenerator
from envs.minimal_jsp_env.util.jsp_solver import JSPSolver


def test_min_entropy():
    
    entropy_distribution = [1]
    instance = RandomJSPGeneratorOperationDistirbution(num_jobs=6, num_operations=6, max_op_duration=9).generate(entropy_distribution)

    assert instance.relative_entropy == 0

def test_max_entropy():

    entropy_distribution = [1/36 for i in range(36)]
    instance = RandomJSPGeneratorOperationDistirbution(num_jobs=6, num_operations=6, max_op_duration=9).generate(entropy_distribution)

    assert instance.relative_entropy == 1

def test_6x6_entropies():
    entropy0_2 = [0.5, 0.5]
    entropy0_3 = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    entropy0_4 = [0.08333333333333333, 0.16666666666666666, 0.25, 0.08333333333333333, 0.4166666666666667]
    entropy0_5 = [0.3333333333333333, 0.16666666666666666, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.16666666666666666]
    entropy0_6 = [0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.16666666666666666, 0.08333333333333333, 0.16666666666666666, 0.16666666666666666, 0.08333333333333333]
    entropy0_7 = [0.05555555555555555, 0.1111111111111111, 0.027777777777777776, 0.05555555555555555, 0.05555555555555555, 0.16666666666666666, 0.16666666666666666, 0.05555555555555555, 0.027777777777777776, 0.027777777777777776, 0.08333333333333333, 0.027777777777777776, 0.05555555555555555, 0.05555555555555555, 0.027777777777777776]
    entropy0_8 = [0.16666666666666666, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.027777777777777776, 0.05555555555555555, 0.027777777777777776, 0.05555555555555555, 0.027777777777777776, 0.027777777777777776, 0.05555555555555555, 0.05555555555555555, 0.027777777777777776, 0.027777777777777776, 0.05555555555555555, 0.027777777777777776, 0.08333333333333333, 0.05555555555555555, 0.08333333333333333, 0.027777777777777776]

    for entropy_val in range(2, 9, 1):
        entropy_distribution = eval(f"entropy0_{entropy_val}")
        instance = RandomJSPGeneratorOperationDistirbution(num_jobs=6, num_operations=6, max_op_duration=9).generate(entropy_distribution)
        assert abs(instance.relative_entropy-0.1*entropy_val)<=0.05

def test_jsp_solver():

    instance = RandomJSPGenerator(num_jobs=6, num_operations=6, max_op_duration=9).generate()
    
    assert type(instance.opt_time) == int or type(instance.opt_time) == float
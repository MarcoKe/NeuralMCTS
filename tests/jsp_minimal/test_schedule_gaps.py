from envs.minimal_jsp_env.entities import Operation
from envs.minimal_jsp_env.reward_functions.schedule_gaps import ScheduleGapsReward
from envs.minimal_jsp_env.jsp_env import JobShopEnv
from envs.minimal_jsp_env.util.jsp_generation.random_generator import RandomJSPGenerator


def test_last_op_in_the_end_single_machine():
    generator = RandomJSPGenerator(1, 1, 1)
    env = JobShopEnv(generator)
    reward = ScheduleGapsReward(env)

    schedule = [[(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2)]]
    time_step = 0
    op_ext = (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2)
    start = max([op_ext[1], time_step])
    end = op_ext[2]

    schedule_gaps = reward.get_idle_times(start, end, time_step, schedule)
    assert schedule_gaps == [0]


def test_last_op_in_between_single_machine():
    generator = RandomJSPGenerator(1, 1, 1)
    env = JobShopEnv(generator)
    reward = ScheduleGapsReward(env)

    schedule = [[(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2),
                 (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 2, 3),
                 (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 4, 6)]]
    time_step = 6
    op_ext = (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=1), 2, 3)
    start = max([op_ext[1], time_step])
    end = op_ext[2]

    schedule_gaps = reward.get_idle_times(start, end, time_step, schedule)
    assert schedule_gaps == [0]


def test_last_op_in_the_end_multiple_machines1():
    generator = RandomJSPGenerator(1, 1, 1)
    env = JobShopEnv(generator)
    reward = ScheduleGapsReward(env)

    schedule = [[(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2)],
                []]
    time_step = 0
    op_ext = (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2)
    start = max([op_ext[1], time_step])
    end = op_ext[2]

    schedule_gaps = reward.get_idle_times(start, end, time_step, schedule)
    assert schedule_gaps == [0, 2]


def test_last_op_in_the_end_multiple_machines2():
    generator = RandomJSPGenerator(1, 1, 1)
    env = JobShopEnv(generator)
    reward = ScheduleGapsReward(env)

    schedule = [[(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2)],
                [(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=1, duration=2), 2, 4)]]
    time_step = 2
    op_ext = (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 2, 4)
    start = max([op_ext[1], time_step])
    end = op_ext[2]

    schedule_gaps = reward.get_idle_times(start, end, time_step, schedule)
    assert schedule_gaps == [2, 0]


def test_last_op_in_the_end_multiple_machines3():
    generator = RandomJSPGenerator(1, 1, 1)
    env = JobShopEnv(generator)
    reward = ScheduleGapsReward(env)

    schedule = [[(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 3)],
                [(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=1, duration=2), 0, 2),
                 (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=1, duration=2), 2, 4)]]
    time_step = 3
    op_ext = (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 2, 4)
    start = max([op_ext[1], time_step])
    end = op_ext[2]

    schedule_gaps = reward.get_idle_times(start, end, time_step, schedule)
    assert schedule_gaps == [1, 0]


def test_last_op_in_the_between_multiple_machines():
    generator = RandomJSPGenerator(1, 1, 1)
    env = JobShopEnv(generator)
    reward = ScheduleGapsReward(env)

    schedule = [[(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2),
                 (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 2, 3),
                 (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 4, 6)],
                [(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=1, duration=2), 0, 2)]]
    time_step = 6
    op_ext = (Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=1), 2, 3)
    start = max([op_ext[1], time_step])
    end = op_ext[2]

    schedule_gaps = reward.get_idle_times(start, end, time_step, schedule)
    assert schedule_gaps == [0, 0]

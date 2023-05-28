from envs.minimal_jsp_env.entities import Operation
from envs.minimal_jsp_env.jsp_model import JobShopModel


def test_left_shift_possible():
    model = JobShopModel()

    machine_schedule = [(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2), (Operation(job_id=1, op_id=0, unique_op_id=1, machine_type=0, duration=2), 4, 6)]
    earliest_start = 0
    op = Operation(job_id=2, op_id=0, unique_op_id=3, machine_type=0, duration=2)

    possible, start_time, index = model._left_shift_possible(earliest_start, machine_schedule, op.duration)
    assert possible
    assert start_time == 2

    machine_schedule.insert(index, (op, start_time, start_time + op.duration))
    assert machine_schedule[1][0].job_id == 2


def test_left_shift_not_possible():
    model = JobShopModel()

    # not possible because gap not big enough
    machine_schedule = [(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2), (Operation(job_id=1, op_id=0, unique_op_id=1, machine_type=0, duration=2), 3, 5)]
    earliest_start = 0
    op_duration = 2

    possible, start_time, index = model._left_shift_possible(earliest_start, machine_schedule, op_duration)
    assert not possible

    # not possible because earliest start too late
    machine_schedule = [(Operation(job_id=0, op_id=0, unique_op_id=0, machine_type=0, duration=2), 0, 2), (Operation(job_id=1, op_id=0, unique_op_id=1, machine_type=0, duration=2), 4, 6)]
    earliest_start = 3

    possible, start_time, index = model._left_shift_possible(earliest_start, machine_schedule, op_duration)
    assert not possible


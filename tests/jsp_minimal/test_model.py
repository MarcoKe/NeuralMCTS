from envs.minimal_jsp_env.entities import Operation
from envs.minimal_jsp_env.jsp_model import JobShopModel


def test_legal_actions():
    schedule = [[] for i in range(6)]
    last_job_ops = [-1 for _ in range(6)]

    remaining_operations = [[],
            [Operation(job_id=1, op_id=0, machine_type=3, duration=8)],
            [Operation(job_id=2, op_id=5, machine_type=1, duration=5)],
            [Operation(job_id=3, op_id=5, machine_type=1, duration=6)],
            [Operation(job_id=4, op_id=0, machine_type=3, duration=6),
             Operation(job_id=4, op_id=5, machine_type=4, duration=8)],
            [Operation(job_id=5, op_id=0, machine_type=3, duration=2),
             Operation(job_id=5, op_id=5, machine_type=4, duration=5)]]

    state = {'remaining_operations': remaining_operations, 'schedule': schedule, 'last_job_ops': last_job_ops}

    model = JobShopModel()
    legal_actions = model.legal_actions(state)
    assert legal_actions == [1, 2, 3, 4, 5]

    state['remaining_operations'] = [[], [], [], [], [], []]
    legal_actions = model.legal_actions(state)
    assert legal_actions == []


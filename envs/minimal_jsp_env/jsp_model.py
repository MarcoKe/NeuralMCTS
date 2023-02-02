from collections import namedtuple

Operation = namedtuple("Operation", ["job_id", "op_id", "machine_type", "duration"])


class JobShopModel:
    @staticmethod
    def _schedule_op(job_id, remaining_operations, schedule):
        op = remaining_operations[job_id].pop(0)
        machine = op.machine_type
        start_time = JobShopModel._determine_start_time(op, schedule)
        schedule[machine].append((op, start_time, start_time + op.duration))
        return remaining_operations, schedule

    @staticmethod
    def _determine_start_time(op: Operation, schedule):
        start_time = 0
        for machine, machine_schedule in enumerate(schedule):
            if len(machine_schedule) > 0:
                last_op, start_time, end_time = machine_schedule[-1]

                # other operation of same job still busy or machine busy
                if last_op.job_id == op.job_id or machine == op.machine_type:
                    if end_time > start_time:
                        start_time = end_time

        return start_time

    @staticmethod
    def _is_done(remaining_operations):
        for j in remaining_operations:
            if len(j) > 0: return False

        return True

    @staticmethod
    def _makespan(schedule):
        makespan = 0

        for machine, machine_schedule in enumerate(schedule):
            if len(machine_schedule) > 0:
                _, _, end_time = machine_schedule[-1]
                if end_time > makespan:
                    makespan = end_time

        return makespan

    @staticmethod
    def step(state, action):
        remaining_ops, schedule = JobShopModel._schedule_op(action, state['remaining_operations'], state['schedule'])

        reward = 0
        done = JobShopModel._is_done(remaining_ops)
        if done:
            reward = JobShopModel._makespan(schedule)
        return {'remaining_operations': remaining_ops, 'schedule': schedule}, reward, done


def print_state(state):
    print(state['remaining_operations'])
    print("\n")
    for machine, machine_schedule in enumerate(state['schedule']):
        print("machine ", machine, " ", machine_schedule)


if __name__ == '__main__':
    num_machines = 3
    num_jobs = 3
    jobs = [[Operation(0, 1, 0, 3), Operation(0, 2, 2, 4), Operation(0, 3, 1, 2)],
            [Operation(1, 1, 1, 3), Operation(1, 2, 2, 4), Operation(1, 3, 0, 2)],
            [Operation(2, 1, 0, 1), Operation(2, 2, 2, 4), Operation(2, 3, 1, 2)]]

    schedule = [[] for i in range(num_machines)]
    remaining_operations = jobs

    model = JobShopModel()

    print(schedule, remaining_operations)
    state = {'remaining_operations': remaining_operations, 'schedule': schedule}
    state, reward, done = model.step(state, 0)
    print_state(state)
    print("\n reward: ", reward, done)

    state, reward, done = model.step(state, 0)
    print_state(state)
    print("\n reward: ", reward, done)

    state, reward, done = model.step(state, 1)
    print_state(state)
    print("\n reward: ", reward, done)

    state, reward, done = model.step(state, 0)
    print_state(state)
    print("\n reward: ", reward, done)

    state, reward, done = model.step(state, 2)
    print_state(state)
    print("\n reward: ", reward, done)

    state, reward, done = model.step(state, 2)
    print_state(state)
    print("\n reward: ", reward, done)

    state, reward, done = model.step(state, 1)
    print_state(state)
    print("\n reward: ", reward, done)

    state, reward, done = model.step(state, 1)
    print_state(state)
    print("\n reward: ", reward, done)

    state, reward, done = model.step(state, 2)
    print_state(state)
    print("\n reward: ", reward, done)

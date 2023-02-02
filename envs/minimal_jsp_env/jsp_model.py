from envs.minimal_jsp_env.entities import Operation
import random


class JobShopModel:
    @staticmethod
    def random_problem(num_jobs, num_machines, max_duration=10):
        remaining_operations = []
        for j in range(num_jobs):
            job = []
            for m in range(num_machines):
                job.append(Operation(j, m, random.randint(0, num_machines-1), random.randint(0, max_duration-1)))

            remaining_operations.append(job)

        schedule = [[] for i in range(num_machines)]

        last_job_ops = [-1 for _ in range(num_jobs)]
        return {'remaining_operations': remaining_operations, 'schedule': schedule, 'last_job_ops': last_job_ops}

    @staticmethod
    def _schedule_op(job_id, remaining_operations, schedule):
        possible = False

        if len(remaining_operations[job_id]) > 0:
            op = remaining_operations[job_id].pop(0)
            machine = op.machine_type
            start_time = JobShopModel._determine_start_time(op, schedule)
            schedule[machine].append((op, start_time, start_time + op.duration))
            possible = True
        return remaining_operations, schedule, possible

    @staticmethod
    def _schedule_op(job_id, remaining_operations, schedule, last_job_ops):
        possible = False

        if len(remaining_operations[job_id]) > 0:
            op = remaining_operations[job_id].pop(0)
            machine = op.machine_type
            start_time = JobShopModel._determine_start_time(op, schedule, last_job_ops)
            schedule[machine].append((op, start_time, start_time + op.duration))
            last_job_ops[op.job_id] = start_time + op.duration
            possible = True
        return remaining_operations, schedule, last_job_ops, possible

    @staticmethod
    def _determine_start_time(op: Operation, schedule, last_job_ops):
        start_time = 0

        if last_job_ops[op.job_id] > 0:
            start_time = last_job_ops[op.job_id]

        machine_schedule = schedule[op.machine_type]
        if len(machine_schedule) > 0:
            last_op, start, end = machine_schedule[-1]

            if end > start_time:
                start_time = end

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
        remaining_ops, schedule, last_job_ops, possible = JobShopModel._schedule_op(action, state['remaining_operations'], state['schedule'], state['last_job_ops'])

        reward = 0
        if not possible: reward = -1
        done = JobShopModel._is_done(remaining_ops)
        if done:
            reward = - JobShopModel._makespan(schedule)
        return {'remaining_operations': remaining_ops, 'schedule': schedule, 'last_job_ops': last_job_ops}, reward, done

def print_state(state):
    print(state['remaining_operations'])
    print("\n")
    for machine, machine_schedule in enumerate(state['schedule']):
        print("machine ", machine, " ", machine_schedule)


if __name__ == '__main__':
    # num_machines = 3
    # num_jobs = 3
    # jobs = [[Operation(0, 1, 0, 3), Operation(0, 2, 2, 4), Operation(0, 3, 1, 2)],
    #         [Operation(1, 1, 1, 3), Operation(1, 2, 2, 4), Operation(1, 3, 0, 2)],
    #         [Operation(2, 1, 0, 1), Operation(2, 2, 2, 4), Operation(2, 3, 1, 2)]]
    #
    # schedule = [[] for i in range(num_machines)]
    # remaining_operations = jobs
    # model = JobShopModel()
    # state = model.random_problem(3, 3)
    # remaining_operations = state['remaining_operations']
    # schedule = state['schedule']
    #
    # print(schedule, remaining_operations)
    # state = {'remaining_operations': remaining_operations, 'schedule': schedule}
    # state, reward, done = model.step(state, 0)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    # state, reward, done = model.step(state, 0)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    # state, reward, done = model.step(state, 1)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    # state, reward, done = model.step(state, 0)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    # state, reward, done = model.step(state, 2)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    # state, reward, done = model.step(state, 2)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    # state, reward, done = model.step(state, 1)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    # state, reward, done = model.step(state, 1)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    # state, reward, done = model.step(state, 2)
    # print_state(state)
    # print("\n reward: ", reward, done)
    #
    #
    # jobs = [[],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [Operation(5, 1, 0, 11)]]
    #
    # schedule = [[], [(Operation(5, 0, 1, 6), 0, 6)], [(Operation(3, 0, 2, 10), 0, 10)], [], [(Operation(3, 1, 4, 10), 10, 20)], []]
    # remaining_operations = jobs
    # model = JobShopModel()
    #
    # print(schedule, remaining_operations)
    # state = {'remaining_operations': remaining_operations, 'schedule': schedule}
    # state, reward, done = model.step(state, 5)
    # print_state(state)
    # print("\n reward: ", reward, done)

    schedule = [[] for i in range(6)]
    last_job_ops = [-1 for _ in range(6)]

    jobs = [[Operation(job_id=0, op_id=0, machine_type=0, duration=10), Operation(job_id=0, op_id=1, machine_type=1, duration=7), Operation(job_id=0, op_id=2, machine_type=2, duration=9), Operation(job_id=0, op_id=3, machine_type=4, duration=7), Operation(job_id=0, op_id=4, machine_type=3, duration=7), Operation(job_id=0, op_id=5, machine_type=5, duration=3)], [Operation(job_id=1, op_id=0, machine_type=3, duration=8), Operation(job_id=1, op_id=1, machine_type=5, duration=8), Operation(job_id=1, op_id=2, machine_type=4, duration=4), Operation(job_id=1, op_id=3, machine_type=2, duration=3), Operation(job_id=1, op_id=4, machine_type=1, duration=3), Operation(job_id=1, op_id=5, machine_type=0, duration=4)], [Operation(job_id=2, op_id=0, machine_type=3, duration=4), Operation(job_id=2, op_id=1, machine_type=4, duration=1), Operation(job_id=2, op_id=2, machine_type=0, duration=8), Operation(job_id=2, op_id=3, machine_type=2, duration=8), Operation(job_id=2, op_id=4, machine_type=5, duration=2), Operation(job_id=2, op_id=5, machine_type=1, duration=5)], [Operation(job_id=3, op_id=0, machine_type=4, duration=10), Operation(job_id=3, op_id=1, machine_type=0, duration=6), Operation(job_id=3, op_id=2, machine_type=2, duration=7), Operation(job_id=3, op_id=3, machine_type=5, duration=8), Operation(job_id=3, op_id=4, machine_type=3, duration=11), Operation(job_id=3, op_id=5, machine_type=1, duration=6)], [Operation(job_id=4, op_id=0, machine_type=3, duration=6), Operation(job_id=4, op_id=1, machine_type=5, duration=7), Operation(job_id=4, op_id=2, machine_type=0, duration=1), Operation(job_id=4, op_id=3, machine_type=2, duration=5), Operation(job_id=4, op_id=4, machine_type=1, duration=9), Operation(job_id=4, op_id=5, machine_type=4, duration=8)], [Operation(job_id=5, op_id=0, machine_type=3, duration=2), Operation(job_id=5, op_id=1, machine_type=0, duration=11), Operation(job_id=5, op_id=2, machine_type=1, duration=4), Operation(job_id=5, op_id=3, machine_type=5, duration=4), Operation(job_id=5, op_id=4, machine_type=2, duration=11), Operation(job_id=5, op_id=5, machine_type=4, duration=5)]]

    remaining_operations = jobs
    state = {'remaining_operations': remaining_operations, 'schedule': schedule, 'last_job_ops': last_job_ops}

    model = JobShopModel()
    state = model.random_problem(3, 3)
    remaining_operations = state['remaining_operations']
    schedule = state['schedule']

    for a in [4, 3, 0, 4, 1, 1, 5, 5, 0, 4, 5, 3, 4, 0, 2, 5, 5, 2, 4, 3, 0, 1, 0, 4, 2, 1, 4, 4, 1, 3, 4, 5, 1, 4, 1, 5, 4, 4, 3, 1, 4, 4, 2, 4, 0, 5, 0, 0, 3, 4, 0, 4, 1, 3, 1, 2, 5, 2]:
        state, reward, done = model.step(state, a)

    print(model._makespan(schedule))

    from envs.minimal_jsp_env.util.visualization.gantt_visualizer import create_gantt

    create_gantt(schedule)

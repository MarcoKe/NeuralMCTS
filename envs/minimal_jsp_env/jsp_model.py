import copy
from envs.model import Model
from envs.minimal_jsp_env.entities import Operation
import random
import numpy as np


class JobShopModel(Model):
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def random_problem(num_jobs, num_machines, max_duration=10):
        remaining_operations = []
        op_id = 0
        for j in range(num_jobs):
            job = []
            for m in range(num_machines):
                job.append(Operation(j, m, op_id, random.randint(0, num_machines-1), random.randint(0, max_duration-1)))
                op_id += 1
            remaining_operations.append(job)

        schedule = [[] for i in range(num_machines)]
        last_job_ops = [-1 for _ in range(num_jobs)]
        durations = np.array([[op.duration for op in job] for job in remaining_operations])
        lower_bounds = np.cumsum(durations, axis=1, dtype=np.single).flatten()

        return {'remaining_operations': remaining_operations, 'schedule': schedule, 'last_job_ops': last_job_ops,
                'init_makespan_estimate': lower_bounds.max()}

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
            possible = True

            op = remaining_operations[job_id].pop(0)
            machine = op.machine_type
            start_time = JobShopModel._last_op_end(last_job_ops, op)
            machine_schedule = schedule[op.machine_type]
            if len(machine_schedule) == 0:
                schedule[machine].append((op, start_time, start_time + op.duration))
                last_job_ops[op.job_id] = start_time + op.duration
                return remaining_operations, schedule, last_job_ops, possible

            left_shift, left_shift_time, insertion_index = JobShopModel._left_shift_possible(start_time, machine_schedule, op.duration)
            if left_shift:
                schedule[machine].insert(insertion_index, (op, left_shift_time, left_shift_time + op.duration))
                new_time = left_shift_time + op.duration
                last_job_ops[op.job_id] = new_time if new_time > last_job_ops[op.job_id] else last_job_ops[op.job_id]

            else:
                last_op, start, end = machine_schedule[-1]

                if end > start_time:
                    start_time = end

                schedule[machine].append((op, start_time, start_time + op.duration))
                last_job_ops[op.job_id] = start_time + op.duration

        return remaining_operations, schedule, last_job_ops, possible

    @staticmethod
    def _left_shift_possible(earliest_start, machine_schedule, op_duration):
        if earliest_start < 0:
            earliest_start = 0

        last_end = earliest_start
        for index, (op, start_time, end_time) in enumerate(machine_schedule):
            if end_time < last_end:
                continue

            if (start_time - last_end) >= op_duration:
                return True, last_end, index

            last_end = end_time

        return False, -1, -1

    @staticmethod
    def _last_op_end(last_job_ops, op: Operation):
        start_time = 0

        if last_job_ops[op.job_id] > 0:
            start_time = last_job_ops[op.job_id]

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
        remaining_ops, schedule, last_job_ops, possible = \
            JobShopModel._schedule_op(action, state['remaining_operations'], state['schedule'], state['last_job_ops'])

        reward = (0, 0)
        if not possible:
            reward = (-1, -1)
        done = JobShopModel._is_done(remaining_ops)
        if done:
            lower_bounds_diff = JobShopModel._makespan(schedule) - state['init_makespan_estimate'].max()
            reward = (- JobShopModel._makespan(schedule), - lower_bounds_diff)

        return {'remaining_operations': remaining_ops, 'schedule': schedule, 'last_job_ops': last_job_ops,
                'init_makespan_estimate': state['init_makespan_estimate']}, reward, done

    @staticmethod
    def legal_actions(state):
        return [job_id for job_id in range(len(state['remaining_operations'])) if len(state['remaining_operations'][job_id]) > 0]


if __name__ == '__main__':
    from envs.minimal_jsp_env.util.visualization.gantt_visualizer import create_gantt
    from envs.minimal_jsp_env.util.jsp_generation.random_generator import RandomJSPGenerator
    gen = RandomJSPGenerator(6, 6, 6)
    from envs.minimal_jsp_env.util.jsp_conversion.samsonov_reader import SamsonovReader
    from envs.gnn_jsp_env.jsp_env import GNNJobShopEnv
    env = GNNJobShopEnv(gen)
    reader = SamsonovReader()
    instance = reader.read_instance('data/jsp_instances/6x6x6/6x6_171_inst.json')
    state2 = env.set_instance(instance)
    model = JobShopModel()
    from envs.gnn_jsp_env.jsp_model import GNNJobShopModel
    model2 = GNNJobShopModel()

    random.seed(1337)
    import time
    start_time_ = time.time()
    steps = 0
    # for _ in range(1000):
    done = False
    # state2 = model2.random_problem(6, 6, 6)
    state = copy.deepcopy(state2)
    state['remaining_operations'] = state['remaining_ops']

    remaining_operations = state['remaining_operations']
    schedule = state['schedule']
    while not done:
        legal_actions = model.legal_actions(state)

        steps += 1
        action = random.choice(legal_actions)
        state, reward, done = model.step(state, action)
        # create_gantt(schedule)

        state2, reward2, done2 = model2.step(state2, action)
        # create_gantt(state2['schedule'])

    duration = time.time() - start_time_
    print("duration: ", duration, " time per step: ", duration / steps)
    print(reward, reward2)

    create_gantt(schedule)
    # create_gantt(state2['schedule'])

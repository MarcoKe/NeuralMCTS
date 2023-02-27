import copy
from envs.model import Model
from envs.minimal_jsp_env.entities import Operation
import random


class JobShopModel(Model):
    def __init__(self, **kwargs):
        pass

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

    @staticmethod
    def legal_actions(state):
        return [job_id for job_id in range(len(state['remaining_operations'])) if len(state['remaining_operations'][job_id]) > 0]


if __name__ == '__main__':
    model = JobShopModel()
    #
    #
    # import time
    # start_time = time.time()
    # steps = 0
    # for _ in range(1000):
    #     done = False
    #     state = model.random_problem(6, 6)
    #     remaining_operations = state['remaining_operations']
    #     schedule = state['schedule']
    #     while not done:
    #         legal_actions = model.legal_actions(state)
    #
    #         steps += 1
    #         action = random.choice(legal_actions)
    #
    #         state, reward, done = model.step(state, action)
    #
    # duration = time.time() - start_time
    # print("duration: ", duration, " time per step: ", duration / steps)
    # print("calls to step: ", model.step.calls)
    # from envs.minimal_jsp_env.util.visualization.gantt_visualizer import create_gantt

    # create_gantt(schedule)

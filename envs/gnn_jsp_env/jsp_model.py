from copy import deepcopy
from envs.model import Model
from envs.gnn_jsp_env.entities import Operation
from envs.gnn_jsp_env.scheduling_utils import get_legal_pos, put_in_the_end, put_in_between, get_op_nbghs, \
    get_end_time_lbs, get_first_col, get_last_col, get_op_by_id
import numpy as np
import random

# Parameters previously taken from the param_parser TODO take from arguments
et_normalize_coef = 1000  # normalizing constant for feature LBs (end time), normalization way: fea/constant


class GNNJobShopModel(Model):
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def random_problem(num_jobs, num_ops_per_job, num_machines, max_duration=10):
        remaining_operations = []
        id = 0
        for i in range(num_jobs):
            job = []
            for j in range(num_ops_per_job):
                job.append(Operation(i, id, random.randint(0, num_machines - 1), random.randint(0, max_duration - 1)))
                id += 1

            remaining_operations.append(job)

        schedule = [[] for _ in range(num_machines)]

        num_ops = num_jobs * num_machines

        # possible operations to choose from next for each job (initialize with the first tasks for each job)
        possible_next_ops = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, 0].astype(np.int64)
        # boolean values indicating whether all operations of a job have been scheduled or not
        mask = np.full(shape=num_jobs, fill_value=0, dtype=bool)
        # number of operations scheduled on each machine
        ops_per_machine = [len([op for job in remaining_operations for op in job if op.machine_type == m]) for m in
                           range(num_machines)]
        # information for each machine: the ids of the operations scheduled on it (in the scheduled order), and the
        # corresponding start and end times
        machine_infos = {m: {'op_ids': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'start_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'end_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32)} for m in
                         range(num_machines)}
        # time at which the last scheduled operation ends for each job
        last_job_ops = [-1 for _ in range(num_jobs)]
        # time at which the last scheduled operation ends on each machine
        last_machine_ops = [-1 for _ in range(num_machines)]

        adj_matrix = GNNJobShopModel.init_adj_matrix(num_ops, num_jobs)
        features = GNNJobShopModel.init_features(remaining_operations)
        jobs = deepcopy(remaining_operations)

        return {'remaining_ops': remaining_operations, 'schedule': schedule, 'machine_infos': machine_infos,
                'last_job_ops': last_job_ops, 'last_mch_ops': last_machine_ops, 'adj_matrix': adj_matrix,
                'features': features, 'possible_next_ops': possible_next_ops, 'mask': mask, 'jobs': jobs}

    @staticmethod
    def _schedule_op(op_id, state):
        possible = False

        legal_actions = GNNJobShopModel._legal_actions(state['possible_next_ops'], state['mask'])
        if len(state['remaining_ops']) > 0 and op_id in legal_actions:
            op = get_op_by_id(op_id, state['remaining_ops'])
            state['remaining_ops'].remove(op)
            start_time, flag = GNNJobShopModel._determine_start_time(op, state['schedule'], state['last_job_ops'],
                                                                     state['last_mch_ops'], state['machine_infos'])
            # insert the operation at the correct position so that the entries remain sorted according to start_time
            state['schedule'][op.machine_type].append((op, start_time, start_time + op.duration))
            state['schedule'][op.machine_type] = sorted(state['schedule'][op.machine_type], key=lambda x: x[1])

            # update state
            if state['last_job_ops'][op.job_id] < start_time + op.duration:
                state['last_job_ops'][op.job_id] = start_time + op.duration
            if state['last_mch_ops'][op.machine_type] < start_time + op.duration:
                state['last_mch_ops'][op.machine_type] = start_time + op.duration
            GNNJobShopModel._update_adj_matrix(state, op, flag)
            GNNJobShopModel._update_features(state, op)

            possible = True

        return state, possible

    @staticmethod
    def _update_adj_matrix(state, op, flag):
        precd, succd = get_op_nbghs(op, state['machine_infos'])
        state['adj_matrix'][op.op_id] = 0
        state['adj_matrix'][op.op_id, op.op_id] = 1
        state['adj_matrix'][op.op_id, precd] = 1
        state['adj_matrix'][succd, op.op_id] = 1
        if op.op_id not in get_first_col(state):
            state['adj_matrix'][op.op_id, op.op_id - 1] = 1
        # remove the old arc when a new operation inserts between two operations
        if flag and precd != op.op_id and succd != op.op_id:
            state['adj_matrix'][succd, precd] = 0

    @staticmethod
    def _update_features(state, op):
        last_col = get_last_col(state)
        if op.op_id not in last_col:
            state['possible_next_ops'][op.op_id // len(state['last_mch_ops'])] += 1  # len(last_mch_ops) = num_machines
        else:
            state['mask'][op.op_id // len(state['last_mch_ops'])] = 1

        lower_bounds = get_end_time_lbs(state['jobs'], state['machine_infos'])
        finished = np.array([f[1] if f[0] != op.op_id else 1 for f in state['features']])
        state['features'] = np.concatenate((lower_bounds.reshape(-1, 1) / et_normalize_coef,
                                            finished.reshape(-1, 1)), axis=1)

    @staticmethod
    def _determine_start_time(op: Operation, schedule, last_job_ops, last_mch_ops, machine_infos):
        job_ready_time = last_job_ops[op.job_id] if last_job_ops[op.job_id] != -1 else 0
        mch_ready_time = last_mch_ops[op.machine_type] if last_mch_ops[op.machine_type] != -1 else 0
        mch_schedule = schedule[op.machine_type]
        # ids of the operations scheduled on the same machine
        op_ids = machine_infos[op.machine_type]['op_ids']
        # start times of the operations scheduled on the same machine
        start_times = machine_infos[op.machine_type]['start_times']
        # start times of the operations scheduled on the same machine
        end_times = machine_infos[op.machine_type]['end_times']
        # whether the operation is scheduled in the end (False) or between already scheduled operations (True)
        flag = False

        # positions between already scheduled operations on the machine required by the operation
        possible_pos = np.where(job_ready_time < start_times)[0]

        if len(possible_pos) == 0:
            # not possible to schedule the operation between other operations -> put in the end
            op_start_time = put_in_the_end(op, job_ready_time, mch_ready_time, op_ids, start_times, end_times)
        else:
            # positions which fit the length of the operation (there is enough time before the next operation)
            legal_pos_idx, legal_pos, possible_pos_end_times = get_legal_pos(op.duration, job_ready_time,
                                                                             possible_pos, start_times, end_times)
            if len(legal_pos) == 0:
                # no position which can fit the operation -> put in the end
                op_start_time = put_in_the_end(op, job_ready_time, mch_ready_time, op_ids, start_times, end_times)
            else:
                # schedule the operation between other operations
                op_start_time = put_in_between(op, legal_pos_idx, legal_pos, possible_pos_end_times,
                                               op_ids, start_times, end_times)
                flag = True

        return op_start_time, flag

    @staticmethod
    def _is_done(remaining_ops):
        for j in remaining_ops:
            if len(j) > 0:
                return False

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
        new_state, possible = GNNJobShopModel._schedule_op(action, deepcopy(state))

        reward = 0
        if not possible:
            reward = -1
        done = GNNJobShopModel._is_done(new_state['remaining_ops'])
        if done:
            reward = - GNNJobShopModel._makespan(new_state['schedule'])

        return new_state, reward, done

    @staticmethod
    def legal_actions(state):
        print("legal actions:", [job_id for job_id in range(len(state['remaining_ops'])) if
                                 len(state['remaining_ops'][job_id]) > 0])
        return [job_id for job_id in range(len(state['remaining_ops'])) if
                len(state['remaining_ops'][job_id]) > 0]

    @staticmethod
    def _legal_actions(possible_next_op, mask):  # TODO unify?
        return [i for i in possible_next_op[np.where(mask == 0)]]

    @staticmethod
    def init_adj_matrix(num_ops, num_jobs):
        # task ids for first column (array containing the first tasks for each job)
        first_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, 0]
        # task ids for last column (array containing the last tasks for each job)
        last_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, -1]

        # conjunctive arcs showing precedence relations between tasks of the same job
        # np array with 1s on the row above the main diagonal and 0s everywhere else
        conj_nei_up_stream = np.eye(num_ops, k=-1, dtype=np.single)
        # np array with 1s on the row below the main diagonal and 0s everywhere else
        conj_nei_low_stream = np.eye(num_ops, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[last_col] = 0

        # self edges for all nodes
        # np array with 1s on the main diagonal and 0s everywhere else
        self_as_nei = np.eye(num_ops, dtype=np.single)

        adj = self_as_nei + conj_nei_up_stream
        return adj

    @staticmethod
    def init_features(jobs):
        durations = np.array([[op.duration for op in job] for job in jobs])
        lower_bounds = np.cumsum(durations, axis=1, dtype=np.single)
        machine_types = np.array([[op.machine_type for op in job] for job in jobs])
        finished_mark = np.zeros_like(machine_types, dtype=np.single)  # 0 for unfinished, 1 for finished

        # node features: normalized end time lower bounds and binary indicator of whether the action has been scheduled
        features = np.concatenate((lower_bounds.reshape(-1, 1) / et_normalize_coef,  # et_normalize_coef default is 1000
                                   finished_mark.reshape(-1, 1)), axis=1)  # 1 if scheduled, 0 otherwise

        return features


if __name__ == '__main__':
    model = GNNJobShopModel()

    # import time
    # start_time = time.time()
    # steps = 0
    # for _ in range(1000):
    #     done = False
    #     state = model.random_problem(2, 2, 2)
    #     remaining_operations = state['remaining_ops']
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
    #
    # create_gantt(schedule)

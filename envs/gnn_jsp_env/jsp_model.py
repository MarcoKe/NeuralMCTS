from copy import deepcopy
from envs.model import Model
from envs.minimal_jsp_env.entities import Operation
from envs.gnn_jsp_env.scheduling_utils import get_legal_pos, put_in_the_end, put_in_between, get_op_nbghs, \
    get_end_time_lbs, get_first_ops
import numpy as np
import random

norm_coeff = 0


class GNNJobShopModel(Model):
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def random_problem(num_jobs, num_ops_per_job, num_machines, max_duration=10):
        remaining_operations = []
        unique_op_id = 0
        for i in range(num_jobs):
            job = []
            for j in range(num_ops_per_job):
                job.append(Operation(i, j, unique_op_id, random.randint(0, num_machines - 1),
                                     random.randint(0, max_duration - 1)))
                unique_op_id += 1

            remaining_operations.append(job)

        schedule = [[] for _ in range(num_machines)]

        num_ops = num_jobs * num_machines

        # Number of operations scheduled on each machine
        ops_per_machine = [len([op for job in remaining_operations for op in job if op.machine_type == m]) for m in
                           range(num_machines)]
        # Information for each machine: the ids of the operations scheduled on it (in the scheduled order), and the
        # corresponding start and end times
        machine_infos = {m: {'op_ids': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'start_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32),
                             'end_times': -1 * np.ones(ops_per_machine[m], dtype=np.int32)} for m in
                         range(num_machines)}
        # Time at which the last scheduled operation ends for each job
        last_job_ops = [-1 for _ in range(num_jobs)]
        # Time at which the last scheduled operation ends on each machine
        last_machine_ops = [-1 for _ in range(num_machines)]

        jobs = deepcopy(remaining_operations)
        adj_matrix = GNNJobShopModel.init_adj_matrix(num_ops, num_jobs)
        features = GNNJobShopModel.init_features(jobs)

        node_states = np.array([1 if i % num_ops_per_job == 0 else 0 for i in range(num_ops)],
                               dtype=np.single)

        return {'remaining_ops': remaining_operations, 'schedule': schedule, 'machine_infos': machine_infos,
                'last_job_ops': last_job_ops, 'last_mch_ops': last_machine_ops, 'adj_matrix': adj_matrix,
                'features': features, 'node_states': node_states, 'jobs': jobs}

    @staticmethod
    def _schedule_op(job_id, state):
        possible = False

        if len(state['remaining_ops'][job_id]) > 0:
            op = state['remaining_ops'][job_id].pop(0)
            start_time, flag = GNNJobShopModel._determine_start_time(op, state['last_job_ops'],
                                                                     state['last_mch_ops'], state['machine_infos'])
            # Insert the operation at the correct position so that the entries remain sorted according to start_time
            state['schedule'][op.machine_type].append((op, start_time, start_time + op.duration))
            state['schedule'][op.machine_type] = sorted(state['schedule'][op.machine_type], key=lambda x: x[1])

            # Update state
            if state['last_job_ops'][op.job_id] < start_time + op.duration:
                state['last_job_ops'][op.job_id] = start_time + op.duration
            if state['last_mch_ops'][op.machine_type] < start_time + op.duration:
                state['last_mch_ops'][op.machine_type] = start_time + op.duration
            GNNJobShopModel._update_adj_matrix(state, op, flag)
            GNNJobShopModel._update_features(state, op)
            GNNJobShopModel._update_node_states(state, op)

            possible = True

        return state, possible

    @staticmethod
    def _update_adj_matrix(state, op, flag):
        # Update the adjacency matrix after a new operation has been scheduled
        pred, succ = get_op_nbghs(op, state['machine_infos'])
        state['adj_matrix'][op.unique_op_id] = 0
        state['adj_matrix'][op.unique_op_id, op.unique_op_id] = 1
        state['adj_matrix'][op.unique_op_id, pred] = 1
        state['adj_matrix'][succ, op.unique_op_id] = 1
        if op.unique_op_id not in get_first_ops(state):
            state['adj_matrix'][op.unique_op_id, op.unique_op_id - 1] = 1
        # Remove the old arc when a new operation inserts between two operations
        if flag and pred != op.unique_op_id and succ != op.unique_op_id:
            state['adj_matrix'][succ, pred] = 0

    @staticmethod
    def _update_features(state, op):
        # Update the operations' features after a new operation has been scheduled
        lower_bounds = get_end_time_lbs(state['jobs'], state['machine_infos'])  # recalculate lower bounds
        finished = np.array([f[1] if i != op.unique_op_id
                             else 1 for i, f in enumerate(state['features'])])  # set op as finished
        assert norm_coeff > 0, "The normalization coefficient has not been initialized"

        state['features'] = np.concatenate((lower_bounds.reshape(-1, 1) / norm_coeff,
                                            finished.reshape(-1, 1)), axis=1)

    @staticmethod
    def _update_node_states(state, op):
        succ = op.unique_op_id + 1 if ((op.unique_op_id + 1) % len(state['jobs'][0]) != 0) else op.unique_op_id
        if succ != op.unique_op_id:
            state['node_states'][op.unique_op_id] = 0  # TODO node_states type changes -> fix
            state['node_states'][succ] = 1  # TODO add -1 condition?

    @staticmethod
    def _determine_start_time(op: Operation, last_job_ops, last_mch_ops, machine_infos):
        job_ready_time = last_job_ops[op.job_id] if last_job_ops[op.job_id] != -1 else 0
        mch_ready_time = last_mch_ops[op.machine_type] if last_mch_ops[op.machine_type] != -1 else 0
        # Whether the operation is scheduled between already scheduled operations (True) or in the end (False)
        flag = False

        # Positions between already scheduled operations on the machine required by the operation
        possible_pos = np.where(job_ready_time < machine_infos[op.machine_type]['start_times'])[0]

        if len(possible_pos) == 0:
            # Not possible to schedule the operation between other operations -> put in the end
            op_start_time = put_in_the_end(op, job_ready_time, mch_ready_time, machine_infos[op.machine_type])
        else:
            # Positions which fit the length of the operation (there is enough time before the next operation)
            legal_pos_idx, legal_pos, possible_pos_end_times = get_legal_pos(op.duration, job_ready_time,
                                                                             possible_pos, machine_infos[op.machine_type])
            if len(legal_pos) == 0:
                # No position which can fit the operation -> put in the end
                op_start_time = put_in_the_end(op, job_ready_time, mch_ready_time, machine_infos[op.machine_type])
            else:
                # Schedule the operation between other operations
                op_start_time = put_in_between(op, legal_pos_idx, legal_pos, possible_pos_end_times,
                                               machine_infos[op.machine_type])
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
    def _get_norm_coeff(max_duration, num_ops_per_job, num_jobs):
        i = 10
        while i < max_duration * num_ops_per_job * num_jobs:
            i *= 10
        return i

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
        return [job_id for job_id in range(len(state['remaining_ops'])) if
                len(state['remaining_ops'][job_id]) > 0]

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
        lower_bounds = np.cumsum(durations, axis=1, dtype=np.single)  # lower bounds of operations' completion times
        machine_types = np.array([[op.machine_type for op in job] for job in jobs])
        finished_mark = np.zeros_like(machine_types, dtype=np.single)  # 0 for unfinished, 1 for finished
        global norm_coeff
        norm_coeff = GNNJobShopModel._get_norm_coeff(max(durations.flatten()), len(jobs[0]), len(jobs))

        # node features: normalized end time lower bounds and binary indicator of whether the action has been scheduled
        features = np.concatenate((lower_bounds.reshape(-1, 1) / norm_coeff,  # normalize the lower bounds
                                   finished_mark.reshape(-1, 1)), axis=1)  # 1 if scheduled, 0 otherwise

        return features


if __name__ == '__main__':
    model = GNNJobShopModel()

    import time
    start_time = time.time()
    steps = 0
    for _ in range(1000):
        done = False
        state = model.random_problem(6, 6, 6)
        remaining_operations = state['remaining_ops']
        schedule = state['schedule']
        while not done:
            legal_actions = model.legal_actions(state)

            steps += 1
            action = random.choice(legal_actions)

            state, reward, done = model.step(state, action)

    duration = time.time() - start_time
    print("duration: ", duration, " time per step: ", duration / steps)
    # from envs.minimal_jsp_env.util.visualization.gantt_visualizer import create_gantt
    #
    # create_gantt(schedule)

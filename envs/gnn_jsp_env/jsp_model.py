from copy import deepcopy
from envs.model import Model
from envs.gnn_jsp_env.entities import Operation
import numpy as np
import random

# Parameters previously taken from the param_parser TODO take from arguments
et_normalize_coef = 1000  # normalizing constant for feature LBs (end time), normalization way: fea/constant


class JobShopModel(Model):
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def random_problem(num_jobs, num_ops_per_job, num_machines, max_duration=10):
        remaining_operations = []
        for j in range(num_jobs):
            job = []
            for m in range(num_ops_per_job):
                job.append(Operation(j, m, random.randint(0, num_machines - 1), random.randint(0, max_duration - 1)))
                # -> TODO op_ids not unique!

            remaining_operations.append(job)

        schedule = [[] for _ in range(num_machines)]

        num_ops = num_jobs * num_machines

        # possible operations to choose from next for each job (initialize with the first tasks for each job)
        possible_next_ops = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, 0].astype(np.int64)
        # boolean values indicating whether all operations of a job have been scheduled or not
        mask = np.full(shape=num_jobs, fill_value=0, dtype=bool)  # TODO fix?
        # start times of operations on each machine
        machine_start_times = -1 * np.ones((num_ops_per_job, num_jobs), dtype=np.int32)
        # operation IDs on each machine
        machine_op_ids = -1 * np.ones((num_ops_per_job, num_jobs), dtype=np.int32)
        # time at which the last scheduled operation ends for each job
        last_job_ops = [-1 for _ in range(num_jobs)]
        # time at which the last scheduled operation ends on each machine
        last_machine_ops = [-1 for _ in range(num_machines)]
        # 2D array with the same shape as the instance jobs, containing operations' end times if they are already
        # scheduled and 0 otherwise
        end_times = np.zeros_like(machine_start_times)

        adj_matrix = init_adj_matrix(num_ops, num_jobs)
        features = init_features(remaining_operations)
        jobs = deepcopy(remaining_operations)

        # return {'remaining_ops': remaining_operations, 'schedule': schedule, 'last_job_ops': last_job_ops}
        return {'remaining_ops': remaining_operations, 'schedule': schedule, 'end_times': end_times,
                'last_job_ops': last_job_ops, 'last_mch_ops': last_machine_ops, 'adj_matrix': adj_matrix,
                'features': features, 'possible_next_ops': possible_next_ops, 'mask': mask, 'jobs': jobs,
                'machine_start_times': machine_start_times, 'machine_op_ids': machine_op_ids}

    @staticmethod
    def _schedule_op(job_id, state):
        possible = False

        if len(state['remaining_ops'][job_id]) > 0 \
                and job_id in JobShopModel._legal_actions(state['possible_next_ops'], state['mask']):
            op = state['remaining_ops'][job_id].pop(0)
            start_time, flag = JobShopModel._determine_start_time(op, state['schedule'], state['last_job_ops'],
                                                                  state['last_mch_ops'], state['machine_op_ids'],
                                                                  state['machine_start_times'])
            # insert the operation at the correct position so that the entries remain sorted according to start_time
            state['schedule'][op.machine_type].append((op, start_time, start_time + op.duration))
            state['schedule'][op.machine_type] = sorted(state['schedule'][op.machine_type], key=lambda x: x[1])

            # update state
            if state['last_job_ops'][op.job_id] < start_time + op.duration:
                state['last_job_ops'][op.job_id] = start_time + op.duration
            if state['last_mch_ops'][op.machine_type] < start_time + op.duration:
                state['last_mch_ops'][op.machine_type] = start_time + op.duration
            JobShopModel._update_adj_matrix(state, op, flag)
            JobShopModel._update_features(state, op)

            possible = True

        return state, possible

    @staticmethod
    def _update_adj_matrix(state, op, flag):
        precd, succd = JobShopModel._get_op_nbghs(op, state['machine_op_ids'])
        state['adj_matrix'][op.op_id] = 0
        state['adj_matrix'][op.op_id, op.op_id] = 1
        state['adj_matrix'][op.op_id, precd] = 1
        state['adj_matrix'][succd, op.op_id] = 1
        if op.op_id not in JobShopModel._get_first_col(state):
            state['adj_matrix'][op.op_id, op.op_id - 1] = 1
        # remove the old arc when a new operation inserts between two operations
        if flag and precd != op.op_id and succd != op.op_id:
            state['adj_matrix'][succd, precd] = 0

    @staticmethod
    def _update_features(state, op):
        if op not in JobShopModel._get_last_col(state):  # TODO fix warning
            state['possible_next_ops'][op.op_id // len(state['last_mch_ops'])] += 1
        else:
            state['mask'][op.op_id // len(state['last_mch_ops'])] = 1
        lower_bounds = JobShopModel._get_end_time_lbs(state['jobs'], state['end_times'])
        finished = np.array([f[1] if f[0] != op.op_id else 1 for f in state['features']])
        state['features'] = np.concatenate((lower_bounds.reshape(-1, 1) / et_normalize_coef,
                                            finished.reshape(-1, 1)), axis=1)

    @staticmethod
    def _get_first_col(state):
        num_ops = len(state['jobs'][0])
        num_jobs = len(state['jobs'])
        first_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, 0]
        return first_col

    @staticmethod
    def _get_last_col(state):
        num_ops = len(state['jobs'][0])
        num_jobs = len(state['jobs'])
        last_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, -1]
        return last_col

    @staticmethod
    def _get_end_time_lbs(jobs, end_times):
        """
        Calculates the end time lower bounds for all operations
        :param jobs: array if jobs, where each job is an array of operations
        :param end_times: 2D array with the same shape as jobs, containing operations' end times if they are already
        scheduled and 0 otherwise
        :returns: np array containing the end time lower bounds of all operations
        """
        durations = np.array([[op.duration for op in job] for job in jobs])
        x, y = get_last_nonzero(end_times, 1, invalid_val=-1)
        durations[np.where(end_times != 0)] = 0
        durations[x, y] = end_times[x, y]
        temp = np.cumsum(durations, axis=1)
        temp[np.where(end_times != 0)] = 0
        ret = end_times + temp
        return ret

    @staticmethod
    def _get_op_nbghs(op, machine_op_ids):
        """
        Finds a given operation's predecessor and successor on the machine where the operation is carried out
        """
        action_coord = np.where(machine_op_ids == op.op_id)

        if action_coord[1].item() > 0:
            pred_id = action_coord[0], action_coord[1] - 1
        else:
            pred_id = action_coord[0], action_coord[1]
        pred = machine_op_ids[pred_id].item()

        if action_coord[1].item() + 1 < machine_op_ids.shape[-1]:
            succ_temp_id = action_coord[0], action_coord[1] + 1
        else:
            succ_temp_id = action_coord[0], action_coord[1]
        succ_temp = machine_op_ids[succ_temp_id].item()
        succ = op.op_id if succ_temp < 0 else succ_temp

        return pred, succ

    @staticmethod
    def _determine_start_time(op: Operation, schedule, last_job_ops, last_mch_ops, machine_op_ids,
                              machine_start_times):
        job_ready_time = last_job_ops[op.job_id] if last_job_ops[op.job_id] != -1 else 0
        mch_ready_time = last_mch_ops[op.machine_type] if last_mch_ops[op.machine_type] != -1 else 0
        # start times of the operations scheduled on the same machine
        start_times = machine_start_times[op.machine_type]
        # ids of the operations scheduled on the same machine
        op_ids = machine_op_ids[op.machine_type]
        # whether the operation is scheduled in the end (False) or between already scheduled operations (True)
        flag = False

        # positions between already scheduled operations on the machine required by the operation
        possible_pos = np.where(job_ready_time < start_times)[0]

        if len(possible_pos) == 0:
            # not possible to schedule the operation between other operations -> put in the end
            op_start_time = JobShopModel._put_in_the_end(op, job_ready_time, mch_ready_time, start_times, op_ids)
        else:
            # positions which fit the length of the operation (there is enough time before the next operation)
            legal_pos_idx, legal_pos, possible_pos_end_times = JobShopModel._get_legal_pos(op.duration, job_ready_time,
                                                                                           schedule, possible_pos,
                                                                                           start_times, op_ids)
            if len(legal_pos) == 0:
                # no position which can fit the operation -> put in the end
                op_start_time = JobShopModel._put_in_the_end(op, job_ready_time, mch_ready_time, start_times, op_ids)
            else:
                # schedule the operation between other operations
                op_start_time = JobShopModel._put_in_between(op, legal_pos_idx, legal_pos, possible_pos_end_times,
                                                             start_times, op_ids)
                flag = True

        return op_start_time, flag

    @staticmethod
    def _put_in_between(op, legal_pos_idx, legal_pos, possible_pos_end_times, start_times, op_ids):
        earliest_idx = legal_pos_idx[0]
        earliest_pos = legal_pos[0]
        start_time = possible_pos_end_times[earliest_idx]
        start_times[:] = np.insert(start_times, earliest_pos, start_time)[:-1]
        op_ids[:] = np.insert(op_ids, earliest_pos, op)[:-1]
        return start_time

    @staticmethod
    def _put_in_the_end(op, job_ready_time, mch_ready_time, start_times, op_ids):
        index = np.where(start_times == -1)[0][0]
        op_start_time = max(job_ready_time, mch_ready_time)
        start_times[index] = op_start_time
        op_ids[index] = op[0]
        return op_start_time

    @staticmethod
    def _get_legal_pos(op_dur, job_ready_time, schedule, possible_pos, start_times, op_ids):
        possible_pos_dur = [op.duration for op in schedule if op.op_id in op_ids[possible_pos]]
        op_before_first_possible_pos = get_op_by_id(possible_pos[0] - 1, [op for op in schedule])
        earliest_start_time = max(job_ready_time,
                                  start_times[possible_pos[0] - 1] + op_before_first_possible_pos.duration)
        possible_pos_end_times = np.append(earliest_start_time, (start_times[possible_pos] + possible_pos_dur))[:-1]
        possible_gaps = start_times[possible_pos] - possible_pos_end_times
        legal_pos_idx = np.where(op_dur <= possible_gaps)[0]
        legal_pos = np.take(possible_pos, legal_pos_idx)
        return legal_pos_idx, legal_pos, possible_pos_end_times

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
        new_state, possible = JobShopModel._schedule_op(action, deepcopy(state))

        reward = 0
        if not possible:
            reward = -1
        done = JobShopModel._is_done(new_state['remaining_ops'])
        if done:
            reward = - JobShopModel._makespan(new_state['schedule'])

        return new_state, reward, done

    @staticmethod
    def legal_actions(state):
        return [job_id for job_id in range(len(state['remaining_ops'])) if
                len(state['remaining_ops'][job_id]) > 0]

    @staticmethod
    def _legal_actions(possible_next_op, mask):  # TODO unify?
        return [i for i in possible_next_op[np.where(mask == 0)]]


def get_last_nonzero(arr, axis, invalid_val=-1):
    """
    Finds the last non-zero elements of a 2D array along a given axis
    :returns: the indices of the last non-zero elements for each 1D subarray
    """
    mask = arr != 0
    # array containing the indices of the last non-zero elements in each subarray
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    # coordinates of the last non-zero elements along the given axis in the 2D array
    y_axis = np.where(mask.any(axis=axis), val, invalid_val)
    x_axis = np.arange(arr.shape[0], dtype=np.int64)
    x_ret = x_axis[y_axis >= 0]
    y_ret = y_axis[y_axis >= 0]
    return x_ret, y_ret


def get_op_by_id(op_id, ops):
    for op in ops:
        if op.op_id == op_id:
            return op


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
    model = JobShopModel()


    import time
    start_time = time.time()
    steps = 0
    for _ in range(1000):
        done = False
        state = model.random_problem(2, 2, 2)
        remaining_operations = state['remaining_ops']
        schedule = state['schedule']
        while not done:
            legal_actions = model.legal_actions(state)

            steps += 1
            action = random.choice(legal_actions)

            state, reward, done = model.step(state, action)

    duration = time.time() - start_time
    print("duration: ", duration, " time per step: ", duration / steps)
    print("calls to step: ", model.step.calls)
    from envs.minimal_jsp_env.util.visualization.gantt_visualizer import create_gantt

    create_gantt(schedule)

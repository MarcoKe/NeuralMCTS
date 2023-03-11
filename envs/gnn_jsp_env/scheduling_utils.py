import numpy as np


def get_op_by_id(op_id, ops):
    for op in ops:
        if op.op_id == op_id:
            return op


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


def get_legal_pos(op_dur, job_ready_time, schedule, possible_pos, start_times, op_ids):
    possible_pos_dur = [op.duration for op in schedule if op.op_id in op_ids[possible_pos]]
    op_before_first_possible_pos = get_op_by_id(possible_pos[0] - 1, [op for op in schedule])
    earliest_start_time = max(job_ready_time,
                              start_times[possible_pos[0] - 1] + op_before_first_possible_pos.duration)
    possible_pos_end_times = np.append(earliest_start_time, (start_times[possible_pos] + possible_pos_dur))[:-1]
    possible_gaps = start_times[possible_pos] - possible_pos_end_times
    legal_pos_idx = np.where(op_dur <= possible_gaps)[0]
    legal_pos = np.take(possible_pos, legal_pos_idx)
    return legal_pos_idx, legal_pos, possible_pos_end_times


def put_in_the_end(op, job_ready_time, mch_ready_time, start_times, op_ids):
    index = np.where(start_times == -1)[0][0]
    op_start_time = max(job_ready_time, mch_ready_time)
    start_times[index] = op_start_time
    op_ids[index] = op[0]
    return op_start_time


def put_in_between(op, legal_pos_idx, legal_pos, possible_pos_end_times, start_times, op_ids):
    earliest_idx = legal_pos_idx[0]
    earliest_pos = legal_pos[0]
    start_time = possible_pos_end_times[earliest_idx]
    start_times[:] = np.insert(start_times, earliest_pos, start_time)[:-1]
    op_ids[:] = np.insert(op_ids, earliest_pos, op)[:-1]
    return start_time


def get_end_time_lbs(jobs, end_times):
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


def get_op_nbghs(op, machine_op_ids):
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


def get_first_col(state):
    num_ops = len(state['jobs'][0])
    num_jobs = len(state['jobs'])
    first_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, 0]
    return first_col


def get_last_col(state):
    num_ops = len(state['jobs'][0])
    num_jobs = len(state['jobs'])
    last_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, -1]
    return last_col

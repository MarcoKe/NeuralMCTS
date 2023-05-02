import numpy as np


def get_legal_pos(op_dur, job_ready_time, possible_pos, mch_infos):
    earliest_start_time = max(job_ready_time, mch_infos['end_times'][possible_pos[0] - 1])
    possible_pos_end_times = np.append(earliest_start_time, mch_infos['end_times'][possible_pos])[:-1]
    possible_gaps = mch_infos['start_times'][possible_pos] - possible_pos_end_times
    legal_pos_idx = np.where(op_dur <= possible_gaps)[0]
    legal_pos = np.take(possible_pos, legal_pos_idx)
    return legal_pos_idx, legal_pos, possible_pos_end_times


def put_in_the_end(op, job_ready_time, mch_ready_time, mch_infos):
    index = np.where(mch_infos['start_times'] == -1)[0][0]
    op_start_time = max(job_ready_time, mch_ready_time)
    mch_infos['op_ids'][index] = op.unique_op_id
    mch_infos['start_times'][index] = op_start_time
    mch_infos['end_times'][index] = op_start_time + op.duration
    return op_start_time


def put_in_between(op, legal_pos_idx, legal_pos, possible_pos_end_times, mch_infos):
    earliest_idx = legal_pos_idx[0]
    earliest_pos = legal_pos[0]
    start_time = possible_pos_end_times[earliest_idx]
    mch_infos['op_ids'][:] = np.insert(mch_infos['op_ids'], earliest_pos, op.unique_op_id)[:-1]
    mch_infos['start_times'][:] = np.insert(mch_infos['start_times'], earliest_pos, start_time)[:-1]
    mch_infos['end_times'][:] = np.insert(mch_infos['end_times'], earliest_pos, start_time + op.duration)[:-1]
    return start_time


def get_end_time_lbs(jobs, machine_infos):
    """
    Calculates the end time lower bounds for all operations
    :param jobs: array if jobs, where each job is an array of operations
    :param machine_infos: dictionary where the keys are machine indices and the values contain
    the ids of the operations scheduled on the machine (in the scheduled order), and the
    corresponding start and end times
    :returns: np array containing the end time lower bounds of all operations
    """
    end_times = [m['end_times'][i] for m in machine_infos.values() for i in range(len(m['end_times']))]
    op_ids = [m['op_ids'][i] for m in machine_infos.values() for i in range(len(m['op_ids']))]
    lbs = -1 * np.ones((len(jobs), len(jobs[0])))

    for i, job in enumerate(jobs):
        for j, op in enumerate(job):
            if op.unique_op_id in op_ids:
                lbs[i][j] = end_times[op_ids.index(op.unique_op_id)]
            elif j > 0:
                lbs[i][j] = lbs[i][j-1] + op.duration
            else:
                lbs[i][j] = op.duration

    return lbs


def get_op_nbghs(op, machine_infos):
    """
    Finds a given operation's predecessor and successor on the machine where the operation is carried out
    """
    for key, value in machine_infos.items():
        if op.unique_op_id in value['op_ids']:
            action_coord = [key, np.where(op.unique_op_id == value['op_ids'])[0][0]]
            break

    if action_coord[1].item() > 0:
        pred_id = action_coord[0], action_coord[1] - 1
    else:
        pred_id = action_coord[0], action_coord[1]
    pred = machine_infos[pred_id[0]]['op_ids'][pred_id[1]]

    if action_coord[1].item() + 1 < machine_infos[action_coord[0]]['op_ids'].shape[-1]:
        succ_temp_id = action_coord[0], action_coord[1] + 1
    else:
        succ_temp_id = action_coord[0], action_coord[1]
    succ_temp = machine_infos[succ_temp_id[0]]['op_ids'][succ_temp_id[1]]
    succ = op.unique_op_id if succ_temp < 0 else succ_temp

    return pred, succ


def get_first_col(state):
    num_ops = len(state['features'])
    num_jobs = len(state['jobs'])
    first_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, 0]
    return first_col


def get_last_col(state):
    num_jobs = len(state['features'])
    num_ops = len(state['jobs'][0]) * num_jobs
    last_col = np.arange(start=0, stop=num_ops, step=1).reshape(num_jobs, -1)[:, -1]
    return last_col

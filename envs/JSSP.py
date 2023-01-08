import gym
import numpy as np
from gym.utils import EzPickle
from param_parser import configs
from permissibleLS import permissibleLeftShift


class JSSP:
    def __init__(self, n_j, n_m):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column (array containing the first tasks for each job)
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column (array containing the last tasks for each job)
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB  # function calculating the end time lower bounds for all operations
        self.getNghbs = getActionNbghs  # function returning the action's predecessor and successor

    def reset(self):
        # JSSP instance (processing times and machine orders for the operations)
        data = self.uni_instance_gen(n_j=self.number_of_jobs, n_m=self.number_of_machines, low=configs.low, high=configs.high)
        self.step_count = 0
        # np array holding the machine order in which each job's operations have to be carried out
        self.m = data[-1]
        # np array holding the durations of each job's operations
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        # record action history
        self.partial_sol_sequence = []
        self.flags = []
        self.posRewards = 0

        # initialize adj matrix
        # np array with 1s on the row above the main diagonal and 0s everywhere else
        # conjunctive arcs showing precedence relations between tasks of the same job
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        # np array with 1s on the row below the main diagonal and 0s everywhere else
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        # np array with 1s on the main diagonal and 0s everywhere else
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)  # self edges for all nodes
        self.adj = self_as_nei + conj_nei_up_stream

        # initialize features
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)  # 0 for unfinished, 1 for finished

        # action (node) features: normalized end time lower bounds and binary indicator of whether the action has been scheduled
        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,  # et_normalize_coef default is 1000
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)  # 1 if scheduled, 0 otherwise
        # initialize feasible omega (the first operations for each job)
        self.omega = self.first_col.astype(np.int64)

        # initialize mask (indicates whether all operations of a job have been scheduled or not?)
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # start time of operations on machines
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)

        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        unscheduled = [tuple(f) for f in fea if f[1] == 0]
        schedule = [tuple(f) for f in fea if f[1] == 1]

        self.state = {'unscheduled': unscheduled, 'schedule': schedule,
                      'adj': self.adj, 'fea': fea, 'omega': self.omega, 'mask': self.mask}

        return self.state

    def step(self, state, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect
        if action not in self.partial_sol_sequence:

            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1  # mark action as finished
            dur_a = self.dur[row, col]
            self.partial_sol_sequence.append(action)

            # UPDATE STATE:
            # permissible left shift (whether the action can be scheduled between already scheduled actions)
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + dur_a

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # adj matrix
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0

        # prepare for return
        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        reward = - (self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()
        unscheduled = [tuple(f) for f in fea if f[1] == 0]
        schedule = [tuple(f) for f in fea if f[1] == 1]
        state = {'unscheduled': unscheduled, 'schedule': schedule,
                  'adj': self.adj, 'fea': fea, 'omega': self.omega, 'mask': self.mask}

        return state, reward, self.done()

    def legal_actions(self, state):
        return [i for i in range(len(state['unscheduled']))]

    def create_obs(self, state):
        unscheduled = state['unscheduled'][:]
        schedule = state['schedule'][:]
        unscheduled.extend([(-1, -1)] * (self.number_of_tasks - len(unscheduled)))
        schedule.extend([(-1, -1)] * (self.number_of_tasks - len(schedule)))

        obs = unscheduled + schedule

        obs = [item for t in obs for item in t]  # flat list of all elements in obs (used as NN input)

        return obs

    def done(self):
        return len(self.partial_sol_sequence) == self.number_of_tasks

    def permute_rows(self, x):
        '''
        :param x: np array
        :return: x with the rows permuted in a random order
        '''
        ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
        ix_j = np.random.sample(x.shape).argsort(axis=1)
        return x[ix_i, ix_j]

    def uni_instance_gen(self, n_j, n_m, low, high):
        """
        Generates random action processing times and orders in which the actions have to go through the machines
        :param n_j: number of operations (actions) in the jobs
        :param n_m: number of machines
        :param low: minimum processing time of an action
        :param high: maximum processing time of an action
        """
        # Set normally distributed random values for the operations' processing times
        times = np.random.randint(low=low, high=high, size=(n_j, n_m))
        # Create np array with n_j rows and entries from 1 to n_m in each row
        # For each job, holds the machine order in which its operations have to be carried out
        machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
        # Permute the row's entries into a random order
        machines = self.permute_rows(machines)
        return times, machines


class JSSPGym(JSSP, gym.Env, EzPickle):
    def __init__(self, n_j, n_m):
        JSSP.__init__(self, n_j, n_m)
        EzPickle.__init__(self)

        self.reset()

    def reset(self):
        self.state = JSSP.reset(self)
        return self.state  # TODO self.create_obs(self.state)

    def step(self, action):
        return JSSP.step(self, self.state, action)  # TODO self.create_obs(self.state)

    def raw_state(self):
        return self.state

    def render(self):
        raise NotImplementedError


def lastNonZero(arr, axis, invalid_val=-1):
    """
    Finds the last non-zero element of an array along a given axis
    :returns: the indices of the last non-zero element
    """
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)
    xAxis = np.arange(arr.shape[0], dtype=np.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet

def calEndTimeLB(temp1, dur_cp):
    """
    Calculates the end time lower bounds for all operations
    :param temp1: actual end times of already scheduled operations
    :param dur_cp: np array containing the durations of all operations
    :returns: np array containing the end time lower bounds
    """
    x, y = lastNonZero(temp1, 1, invalid_val=-1)
    dur_cp[np.where(temp1 != 0)] = 0
    dur_cp[x, y] = temp1[x, y]
    temp2 = np.cumsum(dur_cp, axis=1)
    temp2[np.where(temp1 != 0)] = 0
    ret = temp1+temp2
    return ret

def getActionNbghs(action, opIDsOnMchs):
    """
    Finds a given action's predecessor and successor on the machine where the action is carried out (?)
    :param action
    :param opIDsOnMchs:
    :returns: action's predecessor and successor
    """
    coordAction = np.where(opIDsOnMchs == action)
    precd = opIDsOnMchs[coordAction[0], coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]].item()
    succdTemp = opIDsOnMchs[coordAction[0], coordAction[1] + 1 if coordAction[1].item() + 1 < opIDsOnMchs.shape[-1] else coordAction[1]].item()
    succd = action if succdTemp < 0 else succdTemp
    # precedX = coordAction[0]
    # precedY = coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]
    # succdX = coordAction[0]
    # succdY = coordAction[1] + 1 if coordAction[1].item()+1 < opIDsOnMchs.shape[-1] else coordAction[1]
    return precd, succd

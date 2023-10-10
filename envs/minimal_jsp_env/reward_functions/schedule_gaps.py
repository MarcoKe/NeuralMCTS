import gym


class ScheduleGapsReward(gym.RewardWrapper):
    def __init__(self, env):
        self.env = env
        super(ScheduleGapsReward, self).__init__(env)

    def reward(self, reward):  # todo make tests
        if reward == -1:  # action was not possible
            return reward  # todo check how this influences performance - is it better when left out?

        schedule = self.env.state['schedule']
        time_step = self.env.state['last_time_step']
        op = self.env.state['last_op']

        for m in schedule:  # get the operation's start and end time from the schedule
            if len(m) > 0:
                for o in m:
                    if o[0] == op:
                        op_ext = o
        assert op_ext, "The last operation was not found in the schedule"

        start = max([op_ext[1], time_step])
        end = op_ext[2]
        idle_times = [0] * len(schedule)

        # Calculate gaps induced by the last scheduled operation
        if end > time_step:  # otherwise the gaps are already accounted for
            for idx, machine in enumerate(schedule):
                if len(machine) == 0:
                    idle_times[idx] += (end - start)
                else:
                    last_sch_op_start = machine[-1][1]
                    last_sch_op_end = machine[-1][2]
                    if last_sch_op_start < start and last_sch_op_end < end:  # otherwise the gap is already accounted for
                        idle_times[idx] += (end - max([last_sch_op_end, time_step]))

        reward = (op.duration - sum(idle_times)) / self.env.max_op_duration  # normalization

        return reward

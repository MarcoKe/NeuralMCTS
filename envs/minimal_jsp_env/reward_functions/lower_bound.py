import gym


class LowerBoundDifferenceReward(gym.RewardWrapper):
    def __init__(self, env):
        self.env = env
        super(LowerBoundDifferenceReward, self).__init__(env)

    def reward(self, reward):
        makespan = - reward
        if reward == 0:
            return 0
        elif reward == -1:
            return - 1

        makespan_lb = self.env.instance.makespan_lb

        lower_bound_gap = (makespan - makespan_lb) / makespan_lb

        assert makespan >= makespan_lb, f"weird makespan lower bound gap: {lower_bound_gap}, makespan: {makespan}, " \
                                        f"makespan lower bound: {makespan_lb}"
        return - lower_bound_gap

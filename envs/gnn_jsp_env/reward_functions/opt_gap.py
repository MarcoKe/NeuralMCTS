import gym


class OptimalityGapReward(gym.RewardWrapper):
    def __init__(self, env):
        self.env = env
        super(OptimalityGapReward, self).__init__(env)

    def reward(self, reward):
        makespan = - reward
        if reward == 0:
            return 0
        elif reward == -1:
            return - 1

        optimum = self.env.instance.opt_time
        if not optimum:
            print("Reward function requires precomputed optima")
        optimality_gap = (makespan - optimum) / optimum

        assert makespan >= optimum, f"weird opt gap: {optimality_gap}, makespan: {makespan}, optimum: {optimum}"
        return - optimality_gap


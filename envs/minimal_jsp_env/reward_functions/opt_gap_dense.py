import gym


class OptimalityGapRewardDense(gym.RewardWrapper):
    def __init__(self, env):
        self.env = env
        super(OptimalityGapRewardDense, self).__init__(env)

    def reward(self, reward):
        makespan = self.env.model._makespan(self.env.state['schedule'])

        if reward == -1:
            return - 1

        optimum = self.env.instance.opt_time
        assert optimum, "Reward function requires precomputed optima"

        optimality_gap = (makespan - optimum) / optimum

        if self.env.model._is_done(self.env.state['remaining_operations']):
            assert makespan >= optimum, f"weird opt gap: {optimality_gap}, makespan: {makespan}, optimum: {optimum}"
        return - optimality_gap
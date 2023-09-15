from util.object_factory import ObjectFactory
from envs.minimal_jsp_env.reward_functions.opt_gap import OptimalityGapReward
from envs.minimal_jsp_env.reward_functions.opt_gap_dense import OptimalityGapRewardDense
from envs.minimal_jsp_env.reward_functions.lower_bound import LowerBoundDifferenceReward


class RewardFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


reward_functions = RewardFactory()
reward_functions.register_builder('opt_gap', OptimalityGapReward)
reward_functions.register_builder('opt_gap_dense', OptimalityGapRewardDense)
reward_functions.register_builder('lower_bound', LowerBoundDifferenceReward)

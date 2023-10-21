from util.object_factory import ObjectFactory
from envs.minimal_jsp_env.reward_functions.opt_gap import OptimalityGapReward


class RewardFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


gnn_reward_functions = RewardFactory()
gnn_reward_functions.register_builder('opt_gap', OptimalityGapReward)

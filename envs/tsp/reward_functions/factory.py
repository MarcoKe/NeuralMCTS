from util.object_factory import ObjectFactory
from envs.tsp.reward_functions.naive import NaiveReward


class RewardFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


reward_functions = RewardFactory()
reward_functions.register_builder('naive', NaiveReward)


class AbstractIntrinsicReward():
    def __init__(self):
        pass

    def get_reward(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        return self.get_reward(*args, **kwargs)
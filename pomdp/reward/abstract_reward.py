class AbstractReward:
    def __init__(self, path):
        self.path = path

    def get_reward(self, state, action, done_now):
        pass

    def __call__(self, *args, **kwargs):
        return self.get_reward(*args, **kwargs)
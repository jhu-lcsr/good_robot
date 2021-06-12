
class MovingAverageMeter():

    def __init__(self, N):
        self.last_value = None
        self.avg_value = None
        self.N = float(N)

    def put(self, value):
        if self.avg_value is None:
            self.avg_value = value
        else:
            self.avg_value = self.avg_value * ((self.N - 1) / self.N) + value * (1 / self.N)
        self.last_value = value

    def get(self):
        return self.avg_value if self.avg_value is not None else 0
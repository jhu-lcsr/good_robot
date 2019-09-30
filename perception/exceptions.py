
class SensorUnresponsiveException(Exception):

    def __init__(self, *args, **kwargs):
        super(SensorUnresponsiveException, self).__init__(*args, **kwargs)

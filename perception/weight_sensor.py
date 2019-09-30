"""Wrapper class for weight sensor.
"""
import numpy as np
import rospy
from scipy import signal
import time

from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

class WeightSensor(object):
    """Class for reading from a set of load cells.
    """

    def __init__(self, id_mask='F1804', ntaps=4, debug=False):
        """Initialize the WeightSensor.

        Parameters
        ----------
        id_mask : str
            A template for the first n digits of the device IDs for valid load cells.
        ntaps : int
            Maximum number of samples to perform filtering over.
        debug : bool
            If True, have sensor seem to work normally but just return zeros.
        """
        self._id_mask = id_mask
        self._weight_buffers = []
        self._ntaps = ntaps
        self._debug = debug
        self._filter_coeffs = signal.firwin(ntaps, 0.1)
        self._running = False


    def start(self):
        """Start the sensor.
        """
        if rospy.get_name() == '/unnamed':
            raise ValueError('Weight sensor must be run inside a ros node!')
        self._weight_subscriber = rospy.Subscriber('weight_sensor/weights', Float32MultiArray, self._weights_callback)
        self._running = True


    def stop(self):
        """Stop the sensor.
        """
        if not self._running:
            return
        self._weight_subscriber.unregister()
        self._running = False


    def total_weight(self):
        """Read a weight from the sensor in grams.

        Returns
        -------
        weight : float
            The sensor weight in grams.
        """
        weights = self._raw_weights()
        if weights.shape[1] == 0:
            return 0.0
        elif weights.shape[1] < self._ntaps:
            return np.sum(np.mean(weights, axis=1))
        else:
            return self._filter_coeffs.dot(np.sum(weights, axis=0))

    def individual_weights(self):
        """Read individual weights from the load cells in grams.

        Returns
        -------
        weight : float
            The sensor weight in grams.
        """
        weights = self._raw_weights()
        if weights.shape[1] == 0:
            return np.zeros(weights.shape[0])
        elif weights.shape[1] < self._ntaps:
            return np.mean(weights, axis=1)
        else:
            return weights.dot(self._filter_coeffs)

    def tare(self):
        """Zero out (tare) the sensor.
        """
        if not self._running:
            raise ValueError('Weight sensor is not running!')
        rospy.ServiceProxy('weight_sensor/tare', Empty)()

    def _raw_weights(self):
        """Create a numpy array containing the raw sensor weights.
        """
        if self._debug:
            return np.array([[],[],[],[]])

        if not self._running:
            raise ValueError('Weight sensor is not running!')
        if len(self._weight_buffers) == 0:
            time.sleep(0.3)
            if len(self._weight_buffers) == 0:
                raise ValueError('Weight sensor is not retrieving data!')
        weights = np.array(self._weight_buffers)
        return weights

    def _weights_callback(self, msg):
        """Callback for recording weights from sensor.
        """
        # Read weights
        weights = np.array(msg.data)

        # If needed, initialize indiv_weight_buffers
        if len(self._weight_buffers) == 0:
            self._weight_buffers = [[] for i in range(len(weights))]

        # Record individual weights
        for i, w in enumerate(weights):
            if len(self._weight_buffers[i]) == self._ntaps:
                self._weight_buffers[i].pop(0)
            self._weight_buffers[i].append(w)


    def __del__(self):
        self.stop()


if __name__ == '__main__':
    ws = None
    rospy.init_node('weight_sensor_node', anonymous=True)
    ws = WeightSensor()
    ws.start()
    ws.tare()
    while not rospy.is_shutdown():
        print('{:.2f}'.format(ws.total_weight()))
        time.sleep(0.1)

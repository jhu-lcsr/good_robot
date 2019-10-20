"""
CameraInfo msg type from ROS to a CamearInfo class
Oct 20, 2019
@author: Hongtao Wu
"""

class CameraInfo(object):
    def __init__(self):
        self.height = None
        self.width = None
        self. distortion_model = None
        self.D = None # the distortion parameters, (k1, k2, t1, t2, k3)
        self.K = None # intrinsic camera matrix, [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.R = None # Rectification matrix (stero cameras only)
        self.P = None # Projection/camera matrix [fx', 0, cx', Tx, 0, fy', cy', Ty, 0, 0, 1, 0]

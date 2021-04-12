"""
Calibrate with the ROS package aruco_detect
"""

import rospy
import roslib

from geometry_msgs.msg import Transform

class ROSArUcoCalibrate:

    def __init__(self, aruco_tag_len=0.0795):
        print("Please roslaunch roslaunch aruco_detect aruco_detect.launch before you run!")
        self.aruco_tf_topic = "/fiducial_transforms"
        self._aruco_tf_info_sub = rospy.Subscriber(self.aruco_tf_topic, Transform, self._tfCb)
        self.aruco_tf = None

    def _tfCb(self, tf_msg):
        if tf_msg is None:
            rospy.logwarn("_tfCb: tf_msg is None!")
        
        self.aruco_tf = tf_msg
    
    def get_tf(self):
        aruco_tf = self.aruco_tf
        return aruco_tf


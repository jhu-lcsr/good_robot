from learning.inputs.pose import Pose
from transforms3d import euler, quaternions


class DroneState:
    def __init__(self, image=None, state=None):
        self.image = image
        self.state = state

    def get_pos_2d(self):
        return self.state[0:2]

    def get_pos_3d(self):
        return self.state[0:3]

    def get_cam_pos_3d(self):
        return self.state[9:12]

    def get_cam_rot(self):
        return self.state[12:16]

    def get_rot_euler(self):
        return self.state[3:6]

    def get_depth_image(self):
        return self.image[:, :, 3]

    def get_rgb_image(self):
        return self.image[:, :, 0:3]

    def get_cam_pose(self):
        cam_pos = self.state[9:12]
        cam_rot = self.state[12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def get_drone_pose(self):
        drn_pos = self.get_pos_3d()
        drn_rot_euler = self.get_rot_euler()
        drn_rot_quat = euler.euler2quat(drn_rot_euler[0], drn_rot_euler[1], drn_rot_euler[2])
        pose = Pose(drn_pos, drn_rot_quat)
        return pose

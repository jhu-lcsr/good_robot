"""This file implements a somewhat OpenAI Gym compatible interface for training RL models"""
import math
import random
import time

import numpy as np
from transforms3d import euler
from copy import deepcopy

from data_io.env import load_and_convert_path, convert_pos_from_config, convert_yaw_from_config
from drones.airsim_interface.rate import Rate
from drones.droneController import drone_controller_factory
from drones.rollout_exception import RolloutException
from env_config.generation.generate_random_config import make_config_with_landmark
from geometry import vec_to_yaw
from pomdp.convert_action import unnormalize_action
from pomdp.reward.correct_stop_reward import StopCorrectlyReward
from pomdp.reward.path_field_reward import FollowPathFieldReward
from pomdp.state import DroneState

from parameters.parameter_server import get_current_parameters

from utils.text2speech import say

END_DISTANCE = 1.0
PATH_MAX_DISTANCE = 2.0

class PomdpInterface:

    class EnvException(Exception):
        def __init__(self, message):
            super(PomdpInterface.EnvException, self).__init__(message)

    """Unreal must be running before this is called for now..."""
    def __init__(self, instance_id=0, cv_mode=False, is_real=False):
        self.instance_id = instance_id
        self.env_id = None
        self.params = get_current_parameters()["PomdpInterface"]
        step_interval = self.params["step_interval"]
        flight_height = self.params["flight_height"]
        self.max_horizon = self.params["max_horizon"]
        self.scale = self.params["scale"]
        self.is_real = is_real

        self.drone = drone_controller_factory(simulator=not is_real)(instance=instance_id, flight_height=flight_height)
        rate = step_interval / self.drone.get_real_time_rate()

        self.rate = Rate(rate)
        print("Adjusted rate: " + str(rate), "Step interval: ", step_interval)

        self.segment_path = None

        self.rewards = []
        self.reward_weights = []

        self.instruction_set = None
        self.current_segment_ordinal = None
        self.current_segment_idx = None
        self.cv_mode = cv_mode

        self.seg_start = 0
        self.seg_end = 0
        self.stepcount = 0

        self.instruction_override = None

    def _get_reward(self, state, drone_action, done_now):
        # If we don't have an instruction set, we can't provide a reward
        if self.instruction_set is None:
            raise ValueError("No instruction set: Can't provide a reward.")

        # Obtain the reward from the reward function
        rewards = [w * r(state, drone_action, done_now) for r, w in zip(self.rewards, self.reward_weights)]
        total_reward = sum(rewards)

        return total_reward

    def _is_out_of_bounds(self, state):
        curr_pos = state.get_pos_2d()

        if len(self.segment_path) == 0:
            print("OH NO OH NO OH NO!")
            return True

        distances = np.asarray([np.linalg.norm(p - curr_pos) for p in self.segment_path])
        minidx = np.argmin(distances)
        mindist = distances[minidx]
        #print(f"Idx: {minidx} / {len(self.segment_path)}. Dst to path: {mindist}")
        #done = (mindist > END_DISTANCE and minidx >= len(self.segment_path) - 1) or mindist > PATH_MAX_DISTANCE
        done = mindist > PATH_MAX_DISTANCE
        return done

    def land(self):
        """
        If using the real drone, this causes it to land and disarm
        :return:
        """
        try:
            self.drone.land()
        except RolloutException as e:
            raise PomdpInterface.EnvException("Retry rollout")

    def set_environment(self, env_id, instruction_set=None, fast=False):
        """
        Switch the simulation to env_id. Causes the environment configuration from
        configs/configs/random_config_<env_id>.json to be loaded and landmarks arranged in the simulator
        :param env_id: integer ID
        :param instruction_set: Instruction set to follow for displaying instructions
        :param fast: Set to True to skip a delay at a risk of environment not loading before subsequent function calls
        :return:
        """
        self.env_id = env_id

        try:
            self.drone.set_current_env_id(env_id, self.instance_id)
            self.drone.reset_environment()
        except RolloutException as e:
            raise PomdpInterface.EnvException("Retry rollout")

        # This is necessary to allow the new frame to be rendered with the new pomdp, so that the drone doesn't
        # accidentally see the old pomdp at the start
        if not fast:
            time.sleep(0.1)

        self.instruction_set = instruction_set
        self.stepcount = 0
        self.instruction_override = None

    def set_current_segment(self, seg_idx):
        self.current_segment_ordinal = None
        for i, seg in enumerate(self.instruction_set):
            if seg["seg_idx"] == seg_idx:
                self.current_segment_ordinal = i
        assert self.current_segment_ordinal is not None, f"Requested segment {seg_idx} not found in provided instruction data"
        self.current_segment_idx = seg_idx

        try:
            self.drone.set_current_seg_idx(seg_idx, self.instance_id)
        except RolloutException as e:
            raise PomdpInterface.EnvException("Retry rollout")

        end_idx = self.instruction_set[self.current_segment_ordinal]["end_idx"]
        start_idx = self.instruction_set[self.current_segment_ordinal]["start_idx"]
        full_path = load_and_convert_path(self.env_id)
        self.segment_path = full_path[start_idx:end_idx]
        self.instruction_override = None

        self.stepcount = 0
        if start_idx == end_idx:
            return False

        if self.segment_path is not None:
            self.rewards = [FollowPathFieldReward(self.segment_path), StopCorrectlyReward(self.segment_path)]
            self.reward_weights = [1.0, 1.0]
        return True

    def reset(self, seg_idx=0, landmark_pos=None, random_yaw=0):
        try:
            self.rate.reset()
            self.drone.reset_environment()
            self.stepcount = 0
            start_pos, start_angle = self.get_start_pos(seg_idx, landmark_pos)

            if self.cv_mode:
                start_rpy = start_angle
                self.drone.teleport_3d(start_pos, start_rpy, pos_in_airsim=False)

            else:
                start_yaw = start_angle
                if self.params["randomize_init_pos"]:
                    np.random.seed()
                    yaw_offset = float(np.random.normal(0, self.params["init_yaw_variance"], 1))
                    pos_offset = np.random.normal(0, self.params["init_pos_variance"], 2)
                    #print("offset:", pos_offset, yaw_offset)

                    start_pos = np.asarray(start_pos) + pos_offset
                    start_yaw = start_yaw + yaw_offset
                self.drone.teleport_to(start_pos, start_yaw)

            if self.params.get("voice"):
                cmd = self.get_current_nl_command()
                say(cmd)

            self.drone.rollout_begin(self.get_current_nl_command())
            self.rate.sleep(quiet=True)
            #time.sleep(0.2)
            drone_state, image = self.drone.get_state()

            return DroneState(image, drone_state)
        except RolloutException as e:
            raise PomdpInterface.EnvException("Retry rollout")

    def get_end_pos(self, seg_idx=0, landmark_pos=None):

        seg_ordinal = self.seg_idx_to_ordinal(seg_idx)
        if not self.cv_mode:
            if self.instruction_set:
                end_pos = convert_pos_from_config(self.instruction_set[seg_ordinal]["end_pos"])
                end_angle = convert_yaw_from_config(self.instruction_set[seg_ordinal]["end_yaw"])
            else:
                end_pos = [0, 0, 0]
                end_angle = 0
        return end_pos, end_angle

    def seg_idx_to_ordinal(self, seg_idx):
        for i, instr in enumerate(self.instruction_set):
            if instr["seg_idx"] == seg_idx:
                return i

    def get_start_pos(self, seg_idx=0, landmark_pos=None):

        seg_ordinal = self.seg_idx_to_ordinal(seg_idx)
        # If we are not in CV mode, then we have a path to follow and track reward for following it closely
        if not self.cv_mode:
            if self.instruction_set:
                start_pos = convert_pos_from_config(self.instruction_set[seg_ordinal]["start_pos"])
                start_angle = convert_yaw_from_config(self.instruction_set[seg_ordinal]["start_yaw"])
            else:
                start_pos = [0, 0, 0]
                start_angle = 0

        # If we are in CV mode, there is no path to be followed. Instead we are collecting images of landmarks.
        # Initialize the drone to face the position provided. TODO: Generalize this to other CV uses
        else:
            drone_angle = random.uniform(0, 2 * math.pi)
            drone_dist_mult = random.uniform(0, 1)
            drone_dist = 60 + drone_dist_mult * 300
            drone_pos_dir = np.asarray([math.cos(drone_angle), math.sin(drone_angle)])

            start_pos = landmark_pos + drone_pos_dir * drone_dist
            start_height = random.uniform(-1.5, -2.5)
            start_pos = [start_pos[0], start_pos[1], start_height]

            drone_dir = -drone_pos_dir
            start_yaw = vec_to_yaw(drone_dir)
            start_roll = 0
            start_pitch = 0
            start_angle = [start_roll, start_pitch, start_yaw]
        return start_pos, start_angle

    def get_current_nl_command(self):
        if self.instruction_override:
            return self.instruction_override
        if self.current_segment_ordinal < len(self.instruction_set):
            return self.instruction_set[self.current_segment_ordinal]["instruction"]
        return "FINISHED!"

    def override_instruction(self, instr_str):
        self.instruction_override = instr_str

    def act(self, action):
        # Action
        action = deepcopy(action)
        drone_action = action[0:3]
        # print("Action: ", action)

        raw_action = unnormalize_action(drone_action)
        # print(drone_action, raw_action)
        try:
            self.drone.send_local_velocity_command(raw_action)
        except RolloutException as e:
            raise PomdpInterface.EnvException("Retry rollout")

    def await_env(self):
        self.rate.sleep(quiet=True)

    def observe(self, prev_action):
        try:
            drone_state, image = self.drone.get_state()
            state = DroneState(image, drone_state)
            out_of_bounds = self._is_out_of_bounds(state)
            reward = self._get_reward(state, prev_action, out_of_bounds)

            self.stepcount += 1
            expired = self.stepcount > self.max_horizon

            drone_stop = prev_action[3]
            done = out_of_bounds or drone_stop > 0.5 or expired

            # Actually stop the drone (execute stop-action)
            if done:
                self.drone.send_local_velocity_command([0, 0, 0, 1])
                self.drone.rollout_end()

            # TODO: Respect the done
            return state, reward, done, expired, out_of_bounds
        except RolloutException as e:
            raise PomdpInterface.EnvException("Retry rollout")

    def step(self, action):
        """
        Takes an action, executes it in the simulation and returns the state, reward and done indicator
        :param action: array of length 4: [forward velocity, left velocity, yaw rate, stop probability]
        :return: DroneState object, reward (float), done (bool)
        """
        self.act(action)
        self.await_env()
        return self.observe(action)



    # -----------------------------------------------------------------------------------------------------------------
    # CRAP:

    def reset_to_random_cv_env(self, landmark_name=None):
        config, pos_x, pos_z = make_config_with_landmark(landmark_name)
        self.drone.set_current_env_from_config(config, instance_id=self.instance_id)
        time.sleep(0.2)
        landmark_pos_2d = np.asarray([pos_x, pos_z])
        self.cv_mode = True
        return self.reset(landmark_pos=landmark_pos_2d)

    def snap_birdseye(self, fast=False, small_env=False):
        self.drone.reset_environment()
        # TODO: Check environment dimensions
        if small_env:
            pos_birdseye_as = [2.25, 2.35, -4.92*2 - 0.06 - 0.12]
            rpy_birdseye_as = [-1.3089, 0, 0]   # For 15 deg camera
        else:
            pos_birdseye_as = [15, 15, -25]
            rpy_birdseye_as = [-1.3089, 0, 0]   # For 15 deg camera
        self.drone.teleport_3d(pos_birdseye_as, rpy_birdseye_as, fast=fast)
        _, image = self.drone.get_state(depth=False)
        return image

    def snap_cv(self, pos, quat):
        self.drone.reset_environment()
        pos_birdseye_as = pos
        rpy_birdseye_as = euler.quat2euler(quat)
        self.drone.teleport_3d(pos_birdseye_as, rpy_birdseye_as, pos_in_airsim=True, fast=True)
        time.sleep(0.3)
        self.drone.teleport_3d(pos_birdseye_as, rpy_birdseye_as, pos_in_airsim=True, fast=True)
        time.sleep(0.3)
        _, image = self.drone.get_state(depth=False)
        return image

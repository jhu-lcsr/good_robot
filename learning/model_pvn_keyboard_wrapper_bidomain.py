import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from data_io.models import load_pytorch_model
from data_io.instructions import debug_untokenize_instruction
from learning.models.model_pvn_stage1_bidomain import PVN_Stage1_Bidomain
from learning.models.model_pvn_stage2_bidomain import PVN_Stage2_Bidomain
from learning.models.model_pvn_stage2_actor_critic import PVN_Stage2_ActorCritic
from learning.modules.cuda_module import CudaModule
from learning.modules.map_transformer import MapTransformer
from learning.modules.visitation_softmax import VisitationSoftmax
from learning.inputs.pose import Pose
from learning.inputs.vision import standardize_images, standardize_image
from learning.intrinsic_reward.visitation_reward import VisitationReward
from learning.intrinsic_reward.wd_visitation_and_exploration_reward import WDVisitationAndExplorationReward
from learning.intrinsic_reward.map_coverage_reward import MapCoverageReward
from learning.intrinsic_reward.action_oob_reward import ActionOutOfBoundsReward
from learning.intrinsic_reward.visitation_and_exploration_reward import VisitationAndExplorationReward
from learning.inputs.common import cuda_var
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d

from drones.aero_interface.rviz import RvizInterface

from utils.simple_profiler import SimpleProfiler

import parameters.parameter_server as P

from visualization import Presenter


class PVN_Keyboard_Wrapper_Bidomain(CudaModule):

    def __init__(self, run_name="", model_instance_name="only"):
        super(PVN_Keyboard_Wrapper_Bidomain, self).__init__()
        self.instance_name = model_instance_name
        self.s1_params = P.get_current_parameters()["ModelPVN"]["Stage1"]
        self.wrapper_params = P.get_current_parameters()["PVNWrapper"]
        self.real_drone = P.get_current_parameters()["Setup"]["real_drone"]
        self.rviz = None
        if self.real_drone:
            self.rviz = RvizInterface(
                base_name="/pvn/",
                map_topics=["semantic_map", "visitation_dist"],
                markerarray_topics=["instruction"])

        self.rl = self.wrapper_params["learning_mode"] == "reinforcement_learning"
        self.stage1_visitation_prediction = PVN_Stage1_Bidomain(run_name, model_instance_name)
        self.load_models_from_file()

        #self.spatialsoftmax = SpatialSoftmax2d()
        self.visitation_softmax = VisitationSoftmax()

        self.map_transformer_w_to_r = MapTransformer(
            source_map_size=self.s1_params["global_map_size"],
            dest_map_size=self.s1_params["local_map_size"],
            world_size_m=self.s1_params["world_size_m"],
            world_size_px=self.s1_params["world_size_px"]
        )

        self.prev_instruction = None
        self.start_poses = None
        self.seq_step = 0
        self.log_v_dist_w = None
        self.v_dist_w = None
        self.log_goal_oob_score = None
        self.goal_oob_prob_w = None
        self.map_coverage_w = None
        self.map_uncoverage_w = None

    def load_models_from_file(self):
        if self.wrapper_params.get("stage1_file"):
            print("PVNWrapper: Loading Stage 1")
            load_pytorch_model(self.stage1_visitation_prediction, self.wrapper_params["stage1_file"])

    # Policy state is whatever needs to be updated during RL training.
    # Right now we only update the stage 2 weights.
    def get_policy_state(self):
        return {}

    def set_policy_state(self, state):
        pass

    def init_weights(self):
        self.stage1_visitation_prediction.init_weights()
        self.load_models_from_file()

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        self.stage1_visitation_prediction.cuda(device)
        self.map_transformer_w_to_r.cuda(device)
        return self

    def reset(self):
        self.stage1_visitation_prediction.reset()
        self.prev_instruction = None
        self.start_poses = None
        self.log_v_dist_w = None
        self.v_dist_w = None
        self.log_goal_oob_score = None
        self.goal_oob_prob_w = None
        self.map_coverage_w = None
        self.map_uncoverage_w = None
        self.map_coverage_reward.reset()
        self.visitation_reward.reset()
        self.wd_visitation_and_exploration_reward.reset()

    def start_sequence(self):
        self.seq_step = 0
        self.reset()

    def start_segment_rollout(self):
        self.start_sequence()

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def calc_intrinsic_rewards(self, next_state, action):
        if self.v_dist_w is None or self.map_coverage_w is None:
            raise ValueError("Computing intrinsic reward prior to any rollouts!")
        else:
            states_np = next_state.state[np.newaxis, :]
            states = torch.from_numpy(states_np)
            cam_pos = states[:, 0:12]

            if self.s1_params.get("clip_observability") and self.wrapper_params.get("wasserstein_reward"):
                visitation_reward, stop_reward, exploration_reward = self.wd_visitation_and_exploration_reward(
                    self.v_dist_w, self.goal_oob_prob_w, cam_pos, action)

            elif self.s1_params.get("clip_observability"):
                visitation_reward, stop_reward, exploration_reward = self.visitation_and_exploration_reward(
                    self.v_dist_w, self.goal_oob_prob_w, cam_pos, action)
            else:
                visitation_reward, stop_reward = self.visitation_reward(self.v_dist_w, cam_pos, action)
                exploration_reward = 0.0

            #map_reward = self.map_coverage_reward(self.map_coverage_w)
            #return visitation_reward +  map_reward
            negative_per_step_reward = -0.04
            action_oob_reward = self.action_oob_reward.get_reward(action)

            return {"visitation_reward": visitation_reward,
                    "stop_reward": stop_reward,
                    "exploration_reward": exploration_reward,
                    "negative_per_step_reward": negative_per_step_reward,
                    "action_oob_reward": action_oob_reward}


    def states_to_torch(self, state):
        states_np = state.state[np.newaxis, :]
        images_np = state.image[np.newaxis, :]
        images_np = standardize_images(images_np, out_np=True)
        images_fpv = torch.from_numpy(images_np).float()
        states = torch.from_numpy(states_np)
        return states, images_fpv

    def get_action(self, state, instruction, sample=False, rl_rollout=False):
        """
        Given a DroneState (from PomdpInterface) and instruction, produce a numpy 4D action (x, y, theta, pstop)
        :param state: DroneState object with the raw image from the simulator
        :param instruction: Tokenized instruction given the corpus
        :param sample: (Only applies if self.rl): If true, sample action from action distribution. If False, take most likely action.
        #TODO: Absorb corpus within model
        :return:
        """
        self.eval()

        ACTPROF = False
        actprof = SimpleProfiler(print=ACTPROF, torch_sync=ACTPROF)

        states, images_fpv = self.states_to_torch(state)

        first_step = True
        if instruction == self.prev_instruction:
            first_step = False
        if first_step:
            self.reset()
            self.start_poses = self.cam_poses_from_states(states)
            if self.rviz is not None:
                dbg_instr = "\n".join(Presenter().split_lines(debug_untokenize_instruction(instruction), maxchars=45))
                self.rviz.publish_instruction_text("instruction", dbg_instr)

        self.prev_instruction = instruction
        self.seq_step += 1

        instr_len = [len(instruction)] if instruction is not None else None
        instructions = torch.LongTensor(instruction).unsqueeze(0)
        plan_now = self.seq_step % self.s1_params["plan_every_n_steps"] == 0 or first_step

        # Run stage1 visitation prediction
        # TODO: There's a bug here where we ignore images between planning timesteps. That's why must plan every timestep
        if plan_now or True:
            device = next(self.parameters()).device
            images_fpv = images_fpv.to(device)
            states = states.to(device)
            instructions = instructions.to(device)
            self.start_poses = self.start_poses.to(device)

            actprof.tick("start")
            #print("Planning for: " + debug_untokenize_instruction(list(instructions[0].detach().cpu().numpy())))
            self.log_v_dist_w, v_dist_w_poses, self.log_goal_oob_score, rl_outputs = self.stage1_visitation_prediction(
                images_fpv, states, instructions, instr_len,
                plan=[True], firstseg=[first_step],
                noisy_start_poses=self.start_poses,
                start_poses=self.start_poses,
                select_only=True,
                rl=True
            )
            actprof.tick("stage1")

            self.map_coverage_w = rl_outputs["map_coverage_w"]
            self.map_uncoverage_w = rl_outputs["map_uncoverage_w"]
            self.v_dist_w, self.goal_oob_prob_w = self.visitation_softmax(self.log_v_dist_w, self.log_goal_oob_score)
            if self.rviz:
                v_dist_w_np = self.v_dist_w[0].data.cpu().numpy().transpose(1, 2, 0)
                # expand to 0-1 range
                v_dist_w_np[:, :, 0] /= (np.max(v_dist_w_np[:, :, 0]) + 1e-10)
                v_dist_w_np[:, :, 1] /= (np.max(v_dist_w_np[:, :, 1]) + 1e-10)
                self.rviz.publish_map("visitation_dist", v_dist_w_np,
                                      self.s1_params["world_size_m"])

        # Transform to robot reference frame
        cam_poses = self.cam_poses_from_states(states)
        # Log-distributions CANNOT be transformed - the transformer fills empty space with zeroes, which makes sense for
        # probability distributions, but makes no sense for likelihood scores
        map_coverage_r, _ = self.map_transformer_w_to_r(self.map_coverage_w, None, cam_poses)
        map_uncoverage_r, _ = self.map_transformer_w_to_r(self.map_uncoverage_w, None, cam_poses)
        v_dist_r, r_poses = self.map_transformer_w_to_r(self.v_dist_w, None, cam_poses)

        # Run stage2 action generation
        if self.rl:
            actprof.tick("pipes")
            # If RL, stage 2 outputs distributions over actions (following torch.distributions API)
            xvel_dist, yawrate_dist, stop_dist, value = self.stage2_action_generation(v_dist_r, map_uncoverage_r, eval=True)

            actprof.tick("stage2")
            if sample:
                xvel, yawrate, stop = self.stage2_action_generation.sample_action(xvel_dist, yawrate_dist, stop_dist)
            else:
                xvel, yawrate, stop = self.stage2_action_generation.mode_action(xvel_dist, yawrate_dist, stop_dist)

            actprof.tick("sample")
            xvel_logprob, yawrate_logprob, stop_logprob = self.stage2_action_generation.action_logprob(xvel_dist, yawrate_dist, stop_dist, xvel, yawrate, stop)

            xvel = xvel.detach().cpu().numpy()
            yawrate = yawrate.detach().cpu().numpy()
            stop = stop.detach().cpu().numpy()
            xvel_logprob = xvel_logprob.detach()
            yawrate_logprob = yawrate_logprob.detach()
            stop_logprob = stop_logprob.detach()
            value = value.detach()#.cpu().numpy()

            # Add an empty column for sideways velocity
            act = np.concatenate([xvel, np.zeros(xvel.shape), yawrate, stop])

            # This will be needed to compute rollout statistics later on
            #v_dist_w = self.visitation_softmax(self.log_v_dist_w, self.log_goal_oob_score)

            # Keep all the info we will need later for A2C / PPO training
            # TODO: We assume independence between velocity and stop distributions. Not true, but what ya gonna do?
            rl_data = {
                "policy_input": v_dist_r[0].detach(),
                "v_dist_w": self.v_dist_w[0].detach(),
                "policy_input_b": map_uncoverage_r[0].detach(),
                "value_pred": value[0],
                "xvel": xvel,
                "yawrate": yawrate,
                "stop": stop,
                "xvel_logprob": xvel_logprob,
                "yawrate_logprob": yawrate_logprob,
                "stop_logprob": stop_logprob,
                "action_logprob": xvel_logprob + stop_logprob + yawrate_logprob
            }
            actprof.tick("end")
            actprof.loop()
            actprof.print_stats(1)
            if rl_rollout:
                return act, rl_data
            else:
                return act

        else:
            action = self.stage2_action_generation(v_dist_r, firstseg=[first_step], eval=True)
            output_action = action.squeeze().data.cpu().numpy()
            stop_prob = output_action[3]
            output_stop = 1 if stop_prob > self.s2_params["stop_threshold"] else 0
            output_action[3] = output_stop

            return output_action
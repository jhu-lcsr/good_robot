import os
import time
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils_torch import CrossEntropyLoss2d
from models import PixelNet
from models import reinforcement_net
from models import init_trunk_weights
from scipy import ndimage
import matplotlib.pyplot as plt
import utils
from utils import ACTION_TO_ID
from utils import ID_TO_ACTION
from utils_torch import action_space_argmax

try:
    import ptflops
    from ptflops import get_model_complexity_info
except ImportError:
    print('ptflops is not available, cannot count floating point operations. Try: '
          'pip install --user --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git')
    get_model_complexity_info = None
    ptflops = None


class Trainer(object):
    def __init__(self, method, push_rewards, future_reward_discount,
                 is_testing, snapshot_file, force_cpu, goal_condition_len=0, place=False, pretrained=False,
                 flops=False, network='efficientnet', common_sense=False, show_heightmap=False, place_dilation=0.03,
                 common_sense_backprop=True, trial_reward='spot', num_dilation=0, place_common_sense=True):

        self.heightmap_pixels = 224
        self.buffered_heightmap_pixels = 320
        self.half_heightmap_diff = int((self.buffered_heightmap_pixels - self.heightmap_pixels) / 2)
        self.method = method
        self.place = place
        self.flops = flops
        self.goal_condition_len = goal_condition_len
        self.common_sense = common_sense
        self.place_common_sense = self.common_sense and place_common_sense
        self.common_sense_backprop = common_sense_backprop
        self.show_heightmap = show_heightmap
        self.is_testing = is_testing
        self.place_dilation = place_dilation
        self.trial_reward = trial_reward
        if self.place:
            # # Stacking Reward Schedule
            # reward_schedule = (np.arange(5)**2/(2*np.max(np.arange(5)**2)))+0.75
            # self.push_reward = reward_schedule[0]
            # self.grasp_reward = reward_schedule[1]
            # self.grasp_color_reward = reward_schedule[2]
            # self.place_reward = reward_schedule[3]
            # self.place_color_reward = reward_schedule[4]
            self.push_reward = 0.1
            self.grasp_reward = 1.0
            self.grasp_color_reward = 1.25
            self.place_reward = 1.0
            self.place_color_reward = 1.25
        else:
            # Push Grasp Reward Schedule
            self.push_reward = 0.5
            self.grasp_reward = 1.0
            self.grasp_color_reward = 2.0


        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional classification network for supervised learning
        if self.method == 'reactive':
            self.model = PixelNet(self.use_cuda, goal_condition_len=goal_condition_len, place=place, pretrained=pretrained, network=network, num_dilation=num_dilation)
            # self.model = reinforcement_net(self.use_cuda, goal_condition_len=goal_condition_len, place=place, pretrained=pretrained, network=network)

            # Initialize classification loss
            push_num_classes = 3 # 0 - push, 1 - no change push, 2 - no loss
            push_class_weights = torch.ones(push_num_classes)
            push_class_weights[push_num_classes - 1] = 0
            if self.use_cuda:
                self.push_criterion = CrossEntropyLoss2d(push_class_weights.cuda()).cuda()
            else:
                self.push_criterion = CrossEntropyLoss2d(push_class_weights)
            grasp_num_classes = 3 # 0 - grasp, 1 - failed grasp, 2 - no loss
            grasp_class_weights = torch.ones(grasp_num_classes)
            grasp_class_weights[grasp_num_classes - 1] = 0
            if self.use_cuda:
                self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights.cuda()).cuda()
            else:
                self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights)

            # TODO(hkwon214): added place to test block testing
            if place:
                place_num_classes = 3 # 0 - place, 1 - failed place, 2 - no loss
                place_class_weights = torch.ones(place_num_classes)
                place_class_weights[place_num_classes - 1] = 0
                if self.use_cuda:
                    self.place_criterion = CrossEntropyLoss2d(place_class_weights.cuda()).cuda()
                else:
                    self.place_criterion = CrossEntropyLoss2d(place_class_weights)

        # Fully convolutional Q network for deep reinforcement learning
        elif self.method == 'reinforcement':
            self.model = PixelNet(self.use_cuda, goal_condition_len=goal_condition_len, place=place, pretrained=pretrained, network=network, num_dilation=num_dilation)
            # self.model = reinforcement_net(self.use_cuda, goal_condition_len=goal_condition_len, place=place, pretrained=pretrained, network=network)
            self.push_rewards = push_rewards
            self.future_reward_discount = future_reward_discount

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
            if self.use_cuda:
                self.criterion = self.criterion.cuda()

        self.load_snapshot_file_iteration_log = []
        self.iteration = 0
        # Load pre-trained model
        if snapshot_file:

            # PyTorch v0.4 removes periods in state dict keys, but no backwards compatibility :(
            self.load_snapshot_file(snapshot_file)

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        lr = 1e-4
        momentum = 0.9
        weight_decay = 2e-5
        if is_testing:
            lr = 1e-5
            momentum = 0
            weight_decay = 0
        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # Initialize lists to save execution info and RL variables
        # executed action log includes the action, push grasp or place, and the best pixel index
        self.executed_action_log = []
        self.label_value_log = []
        self.trial_label_value_log = []
        self.reward_value_log = []
        self.trial_reward_value_log = []
        self.predicted_value_log = []
        self.trial_predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []
        self.goal_condition_log = []
        self.trial_log = []
        self.trial_success_log = []
        self.grasp_success_log = []
        self.color_success_log = []
        self.change_detected_log = []
        if place:
            self.stack_height_log = []
            self.partial_stack_success_log = []
            self.place_success_log = []

        # logging imitation actions, imitation action embeddings, executed action embeddings
        self.im_action_log = []
        self.im_action_embed_log = []
        self.executed_action_embed_log = []

    def load_snapshot_file(self, snapshot_file, is_testing=None):
        if is_testing is None:
            is_testing = self.is_testing
        # PyTorch v0.4 removes periods in state dict keys, but no backwards compatibility :(
        if self.use_cuda:
            loaded_snapshot_state_dict = torch.load(snapshot_file)
        else:
            loaded_snapshot_state_dict = torch.load(snapshot_file, map_location=torch.device('cpu'))
        loaded_snapshot_state_dict = OrderedDict([(k.replace('conv.1','conv1'), v) if k.find('conv.1') else (k, v) for k, v in loaded_snapshot_state_dict.items()])
        loaded_snapshot_state_dict = OrderedDict([(k.replace('norm.1','norm1'), v) if k.find('norm.1') else (k, v) for k, v in loaded_snapshot_state_dict.items()])
        loaded_snapshot_state_dict = OrderedDict([(k.replace('conv.2','conv2'), v) if k.find('conv.2') else (k, v) for k, v in loaded_snapshot_state_dict.items()])
        loaded_snapshot_state_dict = OrderedDict([(k.replace('norm.2','norm2'), v) if k.find('norm.2') else (k, v) for k, v in loaded_snapshot_state_dict.items()])
        # TODO(ahundt) use map_device param once updated to pytorch 1.4
        # self.model.load_state_dict(loaded_snapshot_state_dict, strict=is_testing, map_device='cuda' if self.use_cuda else 'cpu')
        self.model.load_state_dict(loaded_snapshot_state_dict, strict=is_testing)
        print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))
        if self.use_cuda:
            self.model = self.model.cuda()
        self.load_snapshot_file_iteration_log.append([self.iteration])
        return len(self.load_snapshot_file_iteration_log)

    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        kwargs = {'delimiter': ' ', 'ndmin': 2}
        self.iteration = int(np.loadtxt(os.path.join(transitions_directory, 'iteration.log.txt'), **kwargs)[0, 0])
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), **kwargs)
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), **kwargs)
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log = self.label_value_log.tolist()
        # self.trial_label_value_log = np.loadtxt(os.path.join(transitions_directory, 'trial-label-value.log.txt'), **kwargs)
        # self.trial_label_value_log = self.trial_label_value_log[0:self.iteration]
        # self.trial_label_value_log = self.trial_label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), **kwargs)
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), **kwargs)
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log = self.reward_value_log.tolist()
        if os.path.exists(os.path.join(transitions_directory, 'trial-reward-value.log.txt')):
            self.trial_reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'trial-reward-value.log.txt'), **kwargs)
            self.trial_reward_value_log = self.trial_reward_value_log[0:self.iteration]
            self.trial_reward_value_log = self.trial_reward_value_log.tolist()
        if os.path.exists(os.path.join(transitions_directory, 'trial-predicted-value.log.txt')):
            self.trial_predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'trial-predicted-value.log.txt'), **kwargs)
            self.trial_predicted_value_log = self.trial_predicted_value_log[0:self.iteration]
            self.trial_predicted_value_log = self.trial_predicted_value_log.tolist()
        if os.path.exists(os.path.join(transitions_directory, 'goal-condition.log.txt')):
            self.goal_condition_log = np.loadtxt(os.path.join(transitions_directory, 'goal-condition.log.txt'), **kwargs)
            self.goal_condition_log = self.goal_condition_log[0:self.iteration]
            self.goal_condition_log = self.goal_condition_log.tolist()
        self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), **kwargs)
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), **kwargs)
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log = self.is_exploit_log.tolist()
        if os.path.exists(os.path.join(transitions_directory, 'clearance.log.txt')):
            self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), **kwargs).astype(np.int64)
            self.clearance_log = self.clearance_log.tolist()
        if os.path.exists(os.path.join(transitions_directory, 'load_snapshot_file_iteration.log.txt')):
            self.load_snapshot_file_iteration_log = np.loadtxt(os.path.join(transitions_directory, 'load_snapshot_file_iteration.log.txt'), **kwargs).astype(np.int64)
            self.load_snapshot_file_iteration_log = self.load_snapshot_file_iteration_log.tolist()
        self.trial_log = np.loadtxt(os.path.join(transitions_directory, 'trial.log.txt'), **kwargs)
        self.trial_log = self.trial_log[0:self.iteration]
        self.trial_log = self.trial_log.tolist()
        self.trial_success_log = np.loadtxt(os.path.join(transitions_directory, 'trial-success.log.txt'), **kwargs)
        self.trial_success_log = self.trial_success_log[0:self.iteration]
        self.trial_success_log = self.trial_success_log.tolist()
        self.grasp_success_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-success.log.txt'), **kwargs)
        self.grasp_success_log = self.grasp_success_log[0:self.iteration]
        self.grasp_success_log = self.grasp_success_log.tolist()
        if os.path.exists(os.path.join(transitions_directory, 'color-success.log.txt')):
            self.color_success_log = np.loadtxt(os.path.join(transitions_directory, 'color-success.log.txt'), **kwargs)
            self.color_success_log = self.color_success_log[0:self.iteration]
            self.color_success_log = self.color_success_log.tolist()
        self.change_detected_log = np.loadtxt(os.path.join(transitions_directory, 'change-detected.log.txt'), **kwargs)
        self.change_detected_log = self.change_detected_log[0:self.iteration]
        self.change_detected_log = self.change_detected_log.tolist()
        if self.place:
            self.stack_height_log = np.loadtxt(os.path.join(transitions_directory, 'stack-height.log.txt'), **kwargs)
            self.stack_height_log = self.stack_height_log[0:self.iteration]
            self.stack_height_log = self.stack_height_log.tolist()
            self.partial_stack_success_log = np.loadtxt(os.path.join(transitions_directory, 'partial-stack-success.log.txt'), **kwargs)
            self.partial_stack_success_log = self.partial_stack_success_log[0:self.iteration]
            self.partial_stack_success_log = self.partial_stack_success_log.tolist()
            self.place_success_log = np.loadtxt(os.path.join(transitions_directory, 'place-success.log.txt'), **kwargs)
            self.place_success_log = self.place_success_log[0:self.iteration]
            self.place_success_log = self.place_success_log.tolist()
        if os.path.exists(os.path.join(transitions_directory, 'trial-reward-value.log.txt')):
            self.trial_reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'trial-reward-value.log.txt'), delimiter=' ')
            self.trial_reward_value_log = self.trial_reward_value_log[0:self.iteration]
            self.trial_reward_value_log.shape = (self.trial_reward_value_log.shape[0], 1)
            self.trial_reward_value_log = self.trial_reward_value_log.tolist()
            if len(self.trial_reward_value_log) < self.iteration:
                self.trial_reward_value_log_update()

    def trial_reward_value_log_update(self, reward=None):
        """
        Apply trial reward to the most recently completed trial.

        reward: the reward algorithm to use. Options are 'spot', and 'discounted'.
        """
        if reward is None:
            reward = self.trial_reward
        # update the reward values for a whole trial, not just recent time steps
        end = int(self.clearance_log[-1][0])
        clearance_length = len(self.clearance_log)

        if end <= len(self.reward_value_log):
            # First entry won't be zero...
            if clearance_length == 1:
                start = 0
            else:
                start = int(self.clearance_log[-2][0])

            new_log_values = []
            future_r = None
            # going backwards in time from most recent to oldest step
            for i in reversed(range(start, end)):

                # load the sample i from the trainer so we can then get the label value
                # [sample_stack_height, sample_primitive_action, sample_grasp_success,
                #  sample_change_detected, sample_push_predictions, sample_grasp_predictions,
                #  next_sample_color_heightmap, next_sample_depth_heightmap, sample_color_success,
                #  exp_goal_condition, sample_place_predictions, sample_place_success, sample_color_heightmap,
                #  sample_depth_heightmap] = self.load_sample(i, logger)

                # load the current reward value
                # sample_push_success = True
                # ignore_me_incorrect_future_reward, current_reward = self.get_label_value(
                #     sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected,
                #     sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap,
                #     sample_color_success, goal_condition=exp_goal_condition, prev_place_predictions=sample_place_predictions,
                #     place_success=sample_place_success, reward_multiplier=reward_multiplier)
                # note, r is a list of size 1, future r is None or a float

                # current timestep rewards were stored in the previous timestep in main.py
                # this is confusing, but we are not modifying the previously written code's behavior to reduce
                # the risks of other bugs cropping up with such a change.
                current_reward = self.reward_value_log[i][0]
                if reward == 'spot':
                    if future_r is None:
                        # Give the final time step its own reward twice.
                        future_r = current_reward / self.future_reward_discount if self.future_reward_discount != 0.0 else 0.0
                    if current_reward > 0:
                        # If a nonzero score was received, the reward propagates
                        future_r = current_reward + self.future_reward_discount * future_r
                        new_log_values.append([future_r])
                    else:
                        # If the reward was zero, propagation is stopped
                        new_log_values.append([current_reward])
                        future_r = current_reward
                elif reward == 'discounted':
                    if future_r is None:
                        future_r = current_reward
                    else:
                        future_r = future_r * self.future_reward_discount
                    new_log_values.append([future_r])
                else:
                    raise ValueError('Unsupported trial_reward schedule: ' + str(reward))

            # stick the reward_value_log on the end in the forward time order
            self.trial_reward_value_log += reversed(new_log_values)
            if len(self.trial_reward_value_log) != len(self.reward_value_log):
                print('trial_reward_value_log_update() past end bug, check the code of trainer.py reward_value_log and trial_reward_value_log')
            # print('self.trial_reward_value_log(): ' + str(self.trial_reward_value_log))
        else:
            print('trial_reward_value_log_update() past end bug, check the code. end: ' +
                  str(end) + ' clearance length: ' + str(clearance_length) +
                  ' reward value log length: ' + str(len(self.reward_value_log)))

    def generate_hist_heightmap(self, valid_depth_heightmap, iteration, logger, history_len=3):
        clearance_inds = np.array(self.clearance_log).flatten()

        # append 1 channel of current timestep depth to depth_heightmap_history
        depth_heightmap_history = [valid_depth_heightmap]
        for i in range(1, history_len):
            # if clearance_inds is empty, we haven't had a reset
            if clearance_inds.shape[0] == 0:
                trial_start = 0

            else:
                # find beginning of current trial (iteration after last reset prior to iteration)
                trial_start = clearance_inds[np.searchsorted(clearance_inds, iteration, side='right') - 1]

            # if we try to load history before beginning of a trial, just repeat initial state
            iter_num = max(iteration - i, trial_start)

            # load img at iter_num
            h_i_path = os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % iter_num)
            h_i = cv2.imread(h_i_path, -1)
            if h_i is None:
                # There was an error loading the image
                print('Warning: Could not load depth heightmap image at the following path, using zeros instead: ' + h_i_path)
                h_i = np.zeros(valid_depth_heightmap.shape)
            h_i = h_i.astype(np.float32)/100000
            depth_heightmap_history.append(h_i)

        return np.stack(depth_heightmap_history, axis=-1)

    def load_sample(self, sample_iteration, logger, depth_channels_history=False, history_len=3):
        """Load the data from disk, and run a forward pass with the current model
        """
        sample_primitive_action_id = self.executed_action_log[sample_iteration][0]

        # Load sample RGB-D heightmap
        sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
        sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
        sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

        # if we are using history, load the last t depth heightmaps, calculate numerical depth, and concatenate
        if depth_channels_history:
            sample_depth_heightmap = self.generate_hist_heightmap(sample_depth_heightmap, sample_iteration, logger)

        else:
            sample_depth_heightmap = np.stack([sample_depth_heightmap] * 3, axis=-1)

        # Compute forward pass with sample
        if self.goal_condition_len > 0:
            exp_goal_condition = [self.goal_condition_log[sample_iteration]]
            next_goal_condition = [self.goal_condition_log[sample_iteration+1]]
        else:
            exp_goal_condition = None
            next_goal_condition = None
            sample_color_success = None

        if self.place:
            # print('place loading stack_height_log sample_iteration: ' + str(sample_iteration) + ' log len: ' + str(len(trainer.stack_height_log)))
            sample_stack_height = int(self.stack_height_log[sample_iteration][0])
            next_stack_height = int(self.stack_height_log[sample_iteration+1][0])
        else:
            # set to 1 because stack height is used as the reward multiplier
            sample_stack_height = 1
            next_stack_height = 1

        sample_push_predictions, sample_grasp_predictions, sample_place_predictions, sample_state_feat, output_prob = self.forward(
            sample_color_heightmap, sample_depth_heightmap, is_volatile=True, goal_condition=exp_goal_condition)

        # TODO(adit98) check if changing suffix rather than changing iteration num for getting future heightmap causes issues
        # Load next sample RGB-D heightmap
        next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration + 1)))
        next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
        next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration + 1)), -1)
        next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000

        # TODO(ahundt) tune sample_reward_value and gamma discount rate?
        sample_place_success = None
        # note that push success is always true in robot.push, and didn't affect get_label_value at the time of writing.
        sample_push_success = True
        sample_change_detected = self.change_detected_log[sample_iteration]
        sample_grasp_success = self.grasp_success_log[sample_iteration]
        if self.place:
            sample_place_success = self.partial_stack_success_log[sample_iteration]
        # in this case grasp_color_task is True
        if exp_goal_condition is not None:
            sample_color_success = self.color_success_log[sample_iteration]
        return sample_stack_height, sample_primitive_action_id, sample_grasp_success, sample_change_detected, sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap, sample_color_success, exp_goal_condition, sample_place_predictions, sample_place_success, sample_color_heightmap, sample_depth_heightmap

    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1, goal_condition=None, keep_action_feat=False, use_demo=False, demo_mask=False):

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2,1], order=0)
        assert color_heightmap_2x.shape == depth_heightmap_2x.shape, print(color_heightmap_2x.shape, depth_heightmap_2x.shape)

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)

        # separate each dim of color heightmap and pad, reconcatenate after
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)

        # separate each dim of depth heightmap and pad, reconcatenate after
        depth_heightmap_2x_r =  np.pad(depth_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        depth_heightmap_2x_r.shape = (depth_heightmap_2x_r.shape[0], depth_heightmap_2x_r.shape[1], 1)
        depth_heightmap_2x_g =  np.pad(depth_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        depth_heightmap_2x_g.shape = (depth_heightmap_2x_g.shape[0], depth_heightmap_2x_g.shape[1], 1)
        depth_heightmap_2x_b =  np.pad(depth_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        depth_heightmap_2x_b.shape = (depth_heightmap_2x_b.shape[0], depth_heightmap_2x_b.shape[1], 1)
        depth_heightmap_2x = np.concatenate((depth_heightmap_2x_r, depth_heightmap_2x_g, depth_heightmap_2x_b), axis=2)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        input_depth_image = depth_heightmap_2x.astype(float)
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)
        if self.flops:
            # sorry for the super random code here, but this is where we will check the
            # floating point operations (flops) counts and parameters counts for now...
            print('input_color_data trainer: ' + str(input_color_data.size()))
            class Wrapper(object):
                custom_params = {'input_color_data': input_color_data, 'input_depth_data': input_depth_data, 'goal_condition': goal_condition}
            def input_constructor(shape):
                return Wrapper.custom_params
            flops, params = get_model_complexity_info(self.model, color_heightmap.shape, as_strings=True, print_per_layer_stat=True, input_constructor=input_constructor)
            print('flops: ' + flops + ' params: ' + params)
            exit(0)

        # Pass input data through model
        output_prob, state_feat, output_prob_feat = self.model.forward(input_color_data, input_depth_data,
                is_volatile, specific_rotation, goal_condition=goal_condition, keep_action_feat=keep_action_feat, use_demo=use_demo)

        # TODO(adit98) remove this part if it no longer makes sense
        # if we are keeping action feat, no softmax
        if keep_action_feat and use_demo:
            softmax = nn.Identity()
            channel_ind = Ellipsis
        else:
            softmax = F.softmax
            channel_ind = 0

        # TODO(adit98) if method is reactive, this will not work, see reinforcement method for correct implementation
        if self.method == 'reactive':
            # Return affordances (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    if keep_action_feat and not use_demo:
                        push_feat = output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                        grasp_feat = output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                        if self.place:
                            place_feat = output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                    int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]

                    push_predictions = softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[:,channel_ind,(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = softmax(output_prob[rotate_idx][1], dim=1).cpu().data.numpy()[:,channel_ind,(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    if self.place:
                        place_predictions = softmax(output_prob[rotate_idx][2], dim=1).cpu().data.numpy()[:,channel_ind,(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    if keep_action_feat and not use_demo:
                        push_feat = np.concatenate((push_feat, output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                        grasp_feat = np.concatenate((grasp_feat, output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                        if self.place:
                            place_feat = np.concatenate((place_feat, output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                    int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

                    push_predictions = np.concatenate((push_predictions, softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[:,channel_ind,(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, softmax(output_prob[rotate_idx][1], dim=1).cpu().data.numpy()[:,channel_ind,(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    if self.place:
                        # TODO(zhe) Shouldn't the following line be using output_prob[rotate_idx][2]?
                        place_predictions = np.concatenate((place_predictions, softmax(output_prob[rotate_idx][1], dim=1).cpu().data.numpy()[:,channel_ind,(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif self.method == 'reinforcement':
            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    if keep_action_feat and not use_demo:
                        push_feat = output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                        grasp_feat = output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                        if self.place:
                            place_feat = output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                    int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                        
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,channel_ind,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,channel_ind,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    if self.place:
                        place_predictions = output_prob[rotate_idx][2].cpu().data.numpy()[:,channel_ind,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    if keep_action_feat and not use_demo:
                        push_feat = np.concatenate((push_feat, output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                        grasp_feat = np.concatenate((grasp_feat, output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                        if self.place:
                            place_feat = np.concatenate((place_feat, output_prob_feat[rotate_idx][0].cpu().data.numpy()[:,:,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),
                                    int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,channel_ind,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,channel_ind,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    if self.place:
                        place_predictions = np.concatenate((place_predictions, output_prob[rotate_idx][2].cpu().data.numpy()[:,channel_ind,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        if not self.place:
            place_predictions = None

        if self.common_sense:
            # TODO(ahundt) "common sense" dynamic action space parameters should be accessible from the command line
            # "common sense" dynamic action space, mask pixels we know cannot lead to progress
            # TODO(zhe) The common_sense_action_space function must also use the language mask, or we can implement a seperate function.
            # process feature masks if we need to return feature masks and final preds
            if keep_action_feat and not use_demo:
                # only mask action feature maps from robot obs if demo_mask is set
                if demo_mask:
                    push_feat, grasp_feat, place_feat = utils.common_sense_action_space_mask(depth_heightmap[:, :, 0],
                            push_feat, grasp_feat, place_feat, self.place_dilation, self.show_heightmap, color_heightmap)
                else:
                    push_feat = np.ma.masked_array(push_feat)
                    grasp_feat = np.ma.masked_array(grasp_feat)
                    if self.place:
                        place_feat = np.ma.masked_array(place_feat)

            # mask action, if we are not in demo or if demo_mask is set
            if not use_demo or demo_mask:
                if self.place:
                    push_predictions, grasp_predictions, masked_place_predictions = utils.common_sense_action_space_mask(depth_heightmap[:, :, 0],
                            push_predictions, grasp_predictions, place_predictions, self.place_dilation, self.show_heightmap, color_heightmap)
                    place_predictions = np.ma.masked_array(place_predictions)
                else:
                    push_predictions, grasp_predictions, masked_place_predictions = utils.common_sense_action_space_mask(depth_heightmap[:, :, 0],
                            push_predictions, grasp_predictions, place_predictions=None, place_dilation=self.place_dilation, show_heightmap=self.show_heightmap, color_heightmap=color_heightmap)

            else:
                # Mask pixels we know cannot lead to progress
                push_predictions = np.ma.masked_array(push_predictions)
                grasp_predictions = np.ma.masked_array(grasp_predictions)
                if self.place:
                    place_predictions = np.ma.masked_array(place_predictions)

        else:
            # Mask pixels we know cannot lead to progress, in this case we don't apply common sense masking
            push_predictions = np.ma.masked_array(push_predictions)
            grasp_predictions = np.ma.masked_array(grasp_predictions)
            if self.place:
                place_predictions = np.ma.masked_array(place_predictions)

        # return components depending on flags
        if keep_action_feat and not use_demo:
            return push_feat, grasp_feat, place_feat, push_predictions, grasp_predictions, place_predictions, state_feat, output_prob

        elif use_demo:
            if self.place_common_sense:
                return push_predictions, grasp_predictions, masked_place_predictions
            else:
                return push_predictions, grasp_predictions, np.ma.masked_array(place_predictions)

        if self.place_common_sense:
            return push_predictions, grasp_predictions, masked_place_predictions, state_feat, output_prob
        else:
            return push_predictions, grasp_predictions, place_predictions, state_feat, output_prob

    def end_trial(self):
        self.clearance_log.append([self.iteration])
        return len(self.clearance_log)

    def num_trials(self):
        return len(self.clearance_log)

    def get_label_value(
            self, primitive_action, push_success, grasp_success, change_detected, prev_push_predictions, prev_grasp_predictions,
            next_color_heightmap, next_depth_heightmap, color_success=None, goal_condition=None, place_success=None,
            prev_place_predictions=None, reward_multiplier=1):

        if self.method == 'reactive':

            # Compute label value
            label_value = 0
            if primitive_action == 'push':
                if not change_detected:
                    label_value = 1
            elif primitive_action == 'grasp':
                if not grasp_success:
                    label_value = 1
            elif primitive_action == 'place':
                if not place_success:
                    label_value = 1

            print('Label value: %d' % (label_value))
            return label_value, label_value

        elif self.method == 'reinforcement':
            # Compute current reward
            current_reward = 0
            if primitive_action == 'push':
                if change_detected:
                    current_reward = self.push_reward * reward_multiplier
            elif primitive_action == 'grasp':
                if color_success is None:
                    if grasp_success:
                        current_reward = self.grasp_reward * reward_multiplier
                elif color_success is not None:
                    # HK add if statement
                    if grasp_success and not color_success:
                        #current_reward = 1.0
                        # TODO(hkwon14): fine tune reward function
                        # current_reward = 0
                        current_reward = self.grasp_reward * reward_multiplier
                    # HK: Color: Compute current reward
                    elif grasp_success and color_success:
                        current_reward = self.grasp_color_reward * reward_multiplier
            # TODO(hkwon214): resume here. think of how to check correct color for place. change 'color success' function so it works with place too
            elif primitive_action == 'place':
                if color_success is None:
                    if place_success:
                        current_reward = self.place_reward * reward_multiplier
                elif color_success is not None:
                    # HK add if statement
                    if place_success and not color_success:
                        #current_reward = 1.0
                        # TODO: fine tune reward function
                        current_reward = self.place_reward * reward_multiplier
                    # HK: Color: Compute current reward
                    elif place_success and color_success:
                        current_reward = self.place_color_reward * reward_multiplier

            # Compute future reward
            if self.place and not change_detected and not grasp_success and not place_success:
                future_reward = 0
            elif not self.place and not change_detected and not grasp_success:
                future_reward = 0
            else:
                next_push_predictions, next_grasp_predictions, next_place_predictions, next_state_feat, output_prob = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True, goal_condition=goal_condition)
                future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))
                if self.place:
                    future_reward = max(future_reward, np.max(next_place_predictions))

                # # Experiment: use Q differences
                # push_predictions_difference = next_push_predictions - prev_push_predictions
                # grasp_predictions_difference = next_grasp_predictions - prev_grasp_predictions
                # future_reward = max(np.max(push_predictions_difference), np.max(grasp_predictions_difference))
            reward_str = 'Trainer.get_label_value(): Current reward: %f Current reward multiplier: %f Predicted Future reward: %f ' % (current_reward, reward_multiplier, future_reward)
            if primitive_action == 'push' and not self.push_rewards:
                expected_reward = self.future_reward_discount * future_reward
                reward_str += 'Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward)
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                reward_str += 'Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward)
            print(reward_str)
            return expected_reward, current_reward


    # TODO(adit98) here is where we need to incorporate imitation loss
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_condition=None, symmetric=False):
        """ Compute labels and backpropagate
        """
        # contactable_regions = None
        # if self.common_sense:
        #     if primitive_action == 'push':
        #         contactable_regions = utils.common_sense_action_failure_heuristic(depth_heightmap, gripper_width=0.04, push_length=0.1)
        #     if primitive_action == 'grasp':
        #         contactable_regions = utils.common_sense_action_failure_heuristic(depth_heightmap)
        #     if primitive_action == 'place':
        #         contactable_regions = utils.common_sense_action_failure_heuristic(depth_heightmap)
        action_id = ACTION_TO_ID[primitive_action]
        if self.show_heightmap:
            # visualize the common sense function results
            # show the heightmap
            f = plt.figure()
            # f.suptitle(str(trainer.iteration))
            f.add_subplot(1,3, 1)
            plt.imshow(depth_heightmap)
            f.add_subplot(1,3, 2)
            # f.add_subplot(1,2, 1)
            # if contactable_regions is not None:
            #     plt.imshow(contactable_regions)
            #     f.add_subplot(1,3, 3)
            # plt.imshow(stuff_count)
            plt.show(block=True)
        if self.method == 'reactive':

            # Compute fill value
            fill_value = 2

            # Compute labels
            label = np.zeros((1,self.buffered_heightmap_pixels,self.buffered_heightmap_pixels)) + fill_value
            action_area = np.zeros((self.heightmap_pixels,self.heightmap_pixels))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((self.heightmap_pixels,self.heightmap_pixels)) + fill_value
            # if self.common_sense:
            #     # all areas where we won't be able to contact anything will have
            #     # value 0 which should indicate no action should be taken at these locations
            #     # TODO(ahundt) double check this is factually correct
            #     tmp_label[contactable_regions < 1] = 1 - contactable_regions
            tmp_label[action_area > 0] = label_value
            label[0,self.half_heightmap_diff:(self.buffered_heightmap_pixels-self.half_heightmap_diff),self.half_heightmap_diff:(self.buffered_heightmap_pixels-self.half_heightmap_diff)] = tmp_label

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'push':
                # loss = self.push_criterion(output_prob[best_pix_ind[0]][0], Variable(torch.from_numpy(label).long().cuda()))

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], goal_condition=goal_condition)

                if self.use_cuda:
                    loss = self.push_criterion(output_prob[0][0], Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.push_criterion(output_prob[0][0], Variable(torch.from_numpy(label).long()))
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                #loss_value = loss.cpu().data.numpy()[0] Commented because the result could be 0 dimensional. Next try/catch will solve that
                loss_value = loss.cpu().data.numpy()

            elif primitive_action == 'grasp':
                # loss = self.grasp_criterion(output_prob[best_pix_ind[0]][1], Variable(torch.from_numpy(label).long().cuda()))
                # loss += self.grasp_criterion(output_prob[(best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations][1], Variable(torch.from_numpy(label).long().cuda()))

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], goal_condition=goal_condition)

                if self.use_cuda:
                    loss = self.grasp_criterion(output_prob[0][1], Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.grasp_criterion(output_prob[0][1], Variable(torch.from_numpy(label).long()))
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                #loss_value += loss.cpu().data.numpy()[0] Commented because the result could be 0 dimensional. Next try/catch will solve that
                loss_value += loss.cpu().data.numpy()

                if symmetric and not self.place:
                    # Since grasping can be symmetric when not placing, depending on the robot kinematics,
                    # train with another forward pass of opposite rotation angle
                    opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

                    push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx, goal_condition=goal_condition)

                    if self.use_cuda:
                        loss = self.grasp_criterion(output_prob[0][1], Variable(torch.from_numpy(label).long().cuda()))
                    else:
                        loss = self.grasp_criterion(output_prob[0][1], Variable(torch.from_numpy(label).long()))
                    loss.backward()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    #loss_value += loss.cpu().data.numpy()[0] Commented because the result could be 0 dimensional. Next try/catch will solve that
                    loss_value += loss.cpu().data.numpy()

                    loss_value = loss_value/2

            elif primitive_action == 'place':
                # Note that placing is definitely not symmetric, because an off-center grasp will lead to two different oriented place actions.
                # loss = self.grasp_criterion(output_prob[best_pix_ind[0]][1], Variable(torch.from_numpy(label).long().cuda()))
                # loss += self.grasp_criterion(output_prob[(best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations][1], Variable(torch.from_numpy(label).long().cuda()))

                # Do forward pass with specified rotation (to save gradients)
                push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], goal_condition=goal_condition)

                if self.use_cuda:
                    loss = self.place_criterion(output_prob[0][2], Variable(torch.from_numpy(label).long().cuda()))
                else:
                    loss = self.place_criterion(output_prob[0][2], Variable(torch.from_numpy(label).long()))
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                #loss_value += loss.cpu().data.numpy()[0] Commented because the result could be 0 dimensional. Next try/catch will solve that
                loss_value += loss.cpu().data.numpy()

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()

        elif self.method == 'reinforcement':
            self.optimizer.zero_grad()
            # Compute labels
            label = np.zeros((1,self.buffered_heightmap_pixels,self.buffered_heightmap_pixels))
            action_area = np.zeros((self.heightmap_pixels,self.heightmap_pixels))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((self.heightmap_pixels,self.heightmap_pixels))
            tmp_label[action_area > 0] = label_value
            # these are the label values, mostly consisting of zeros, except for where the robot really went which is at best_pix_ind.
            label[0,self.half_heightmap_diff:(self.buffered_heightmap_pixels-self.half_heightmap_diff),self.half_heightmap_diff:(self.buffered_heightmap_pixels-self.half_heightmap_diff)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((self.heightmap_pixels,self.heightmap_pixels))
            tmp_label_weights[action_area > 0] = 1

            # Do forward pass with specified rotation (to save gradients)
            push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], goal_condition=goal_condition)
            if self.common_sense and self.common_sense_backprop:
                # If the current argmax is masked, the geometry indicates the action would not contact anything.
                # Therefore, we know the action would fail so train the argmax value with 0 reward.
                # This new common sense reward will have the same weight as the actual historically executed action.
                new_best_pix_ind, each_action_max_coordinate, predicted_value = action_space_argmax(primitive_action, push_predictions, grasp_predictions, place_predictions)
                predictions = {0:push_predictions, 1: grasp_predictions, 2: place_predictions}
                if predictions[action_id].mask[each_action_max_coordinate[primitive_action]]:
                    # The tmp_label value will already be 0, so just set the weight.
                    tmp_label_weights[each_action_max_coordinate[primitive_action]] = 1

            # In the commented code we tried to apply Q values at every filtered pixel, but that didn't work well.
            # if self.common_sense:
            #     # all areas where we won't be able to contact anything will have
            #     # mask value 1 which allows the label value zero to be applied
            #     tmp_label_weights = 1 - contactable_regions
            #     # The real robot label gets weight equal to the summ of all heuristic labels, or 1
            #     tmp_label_weights[action_area > 0] = max(np.sum(tmp_label_weights), 1)
            # else:
            #     tmp_label_weights[action_area > 0] = 1
            #     # since we are now taking the mean loss, in this case we switch to the size of tmp_label_weights to counteract dividing by the number of entries
            #     # tmp_label_weights[action_area > 0] = max(tmp_label_weights.size, 1)
            label_weights[0,self.half_heightmap_diff:(self.buffered_heightmap_pixels-self.half_heightmap_diff),self.half_heightmap_diff:(self.buffered_heightmap_pixels-self.half_heightmap_diff)] = tmp_label_weights

            loss_value = 0
            # Compute loss and backward pass

            if self.use_cuda:
                loss = self.criterion(output_prob[0][action_id].view(1,self.buffered_heightmap_pixels,self.buffered_heightmap_pixels), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(output_prob[0][action_id].view(1,self.buffered_heightmap_pixels,self.buffered_heightmap_pixels), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
            loss = loss.sum()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            loss_value = loss.cpu().data.numpy()

            if symmetric and primitive_action == 'grasp' and not self.place:
                # Since grasping can be symmetric when not placing, depending on the robot kinematics,
                # train with another forward pass of opposite rotation angle
                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

                push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx, goal_condition=goal_condition)

                if self.use_cuda:
                    loss = self.criterion(output_prob[0][action_id].view(1,self.buffered_heightmap_pixels,self.buffered_heightmap_pixels), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(output_prob[0][action_id].view(1,self.buffered_heightmap_pixels,self.buffered_heightmap_pixels), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

                loss = loss.sum()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                loss_value = loss.cpu().data.numpy()

                loss_value = loss_value/2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()


    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind, scale_factor=8):
        # TODO(ahundt) once the reward function is back in the 0 to 1 range, make the scale factor 1 again
        canvas = None
        num_rotations = predictions.shape[0]
        # predictions are a masked arrray, so masked regions have the fill value 0
        predictions = predictions.filled(0.0)
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                # prediction_vis[prediction_vis < 0] = 0 # assume probability
                # prediction_vis[prediction_vis > 1] = 1 # assume probability
                # Reduce the dynamic range so the visualization looks better
                prediction_vis = prediction_vis/(np.max(prediction_vis) + 0.00000001)
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (221,211,238), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas

    def randomize_trunk_weights(self, backprop_enabled=None, random_trunk_weights_max=6, random_trunk_weights_reset_iters=10, min_success=2):
        """ Automatically re-initialize the trunk weights until we get something useful.
        """
        if self.is_testing or self.iteration > random_trunk_weights_max * random_trunk_weights_reset_iters:
            # enable backprop
            backprop_enabled = {'push': True, 'grasp': True}
            if self.place:
                backprop_enabled['place'] = True
            return backprop_enabled
        if backprop_enabled is None:
            backprop_enabled = {'push': False, 'grasp': False}
            if self.place:
                backprop_enabled['place'] = False
        if self.iteration < 2:
            return backprop_enabled
        # models_ready_for_backprop = 0
        # executed_action_log includes the action, push grasp or place, and the best pixel index
        max_iteration = np.min([len(self.executed_action_log), len(self.change_detected_log)])
        min_iteration = max(max_iteration - random_trunk_weights_reset_iters, 1)
        actions = np.asarray(self.executed_action_log)[min_iteration:max_iteration, 0]
        successful_push_actions = np.argwhere(np.logical_and(np.asarray(self.change_detected_log)[min_iteration:max_iteration, 0] == 1, actions == ACTION_TO_ID['push']))

        time_to_reset = self.iteration > 1 and self.iteration % random_trunk_weights_reset_iters == 0
        # we need to return if we should backprop
        if (len(successful_push_actions) >= min_success):
            backprop_enabled['push'] = True
        elif not backprop_enabled['grasp'] and time_to_reset:
                init_trunk_weights(self.model, 'push-')

        if (np.sum(np.asarray(self.grasp_success_log)[min_iteration:max_iteration, 0]) >= min_success):
            backprop_enabled['grasp'] = True
        elif not backprop_enabled['grasp'] and time_to_reset:
                init_trunk_weights(self.model, 'grasp-')

        if self.place:
            if np.sum(np.asarray(self.partial_stack_success_log)[min_iteration:max_iteration, 0]) >= min_success:
                backprop_enabled['place'] = True
            elif not backprop_enabled['place'] and time_to_reset:
                init_trunk_weights(self.model, 'place-')
        return backprop_enabled

    def push_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) - rotated_heightmap > 0.02] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_push_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_push_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                push_predictions = tmp_push_predictions
            else:
                push_predictions = np.concatenate((push_predictions, tmp_push_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        return best_pix_ind


    def grasp_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind


    def place_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_place_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_place_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                place_predictions = tmp_place_predictions
            else:
                place_predictions = np.concatenate((place_predictions, tmp_place_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(place_predictions), place_predictions.shape)
        return best_pix_ind


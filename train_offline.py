from demo import Demonstration, load_all_demos
from utils import ACTION_TO_ID
from trainer import Trainer
from tqdm import tqdm
import os
import argparse
import numpy as np
import torch

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--base_model', required=True)
    parser.add_argument('-d', '--demo_dir', required=True, help='path to dir with demos')
    parser.add_argument('-i', '--iterations', default=250, type=int, help='how many training steps')
    parser.add_argument('-s', '--seed', default=1234, type=int)
    parser.add_argument('-t', '--task_type', default='stack', help='stack/row/unstack/vertical_square')
    parser.add_argument('-o', '--out_dir', default=None, help='where to write finetuned model, WILL NOT SAVE IF BLANK')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.65)
    args = parser.parse_args()

    # define workspace_limits (Cols: min max, Rows: x y z (define workspace limits in robot coordinates))
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])

    # seed np.random
    np.random.seed(args.seed)

    # first load the demo(s)
    demos = load_all_demos(args.demo_dir, check_z_height=False,
            task_type=args.task_type)
    num_demos = len(demos)


    # now load the trainer
    model_path = os.path.abspath(args.base_model)
    place_common_sense = (args.task_type != 'unstack')
    if args.task_type == 'stack':
        place_dilation = 0.00
    else:
        place_dilation = 0.05

    trainer = Trainer(method='reinforcement', push_rewards=False,
            future_reward_discount=0.65, is_testing=False, snapshot_file=args.base_model,
            force_cpu=False, goal_condition_len=0, place=True, pretrained=True,
            flops=False, network='densenet', common_sense=True,
            place_common_sense=place_common_sense, show_heightmap=False,
            place_dilation=place_dilation, common_sense_backprop=True,
            trial_reward='discounted', num_dilation=0)

    # next compute the rewards for the trial (all steps successful)
    prog_rewards = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    # compute trial_rewards
    trial_rewards = prog_rewards.copy()
    for i in reversed(range(len(trial_rewards))):
        if i == len(trial_rewards) - 1:
            trial_rewards[i] *= 2
            continue

        trial_rewards[i] += args.future_reward_discount * trial_rewards[i+1]

    # store losses, checkpoint model every 25 iterations
    losses = []
    models = {} # dict {iter: model_weights}

    for i in tqdm(range(args.iterations)):
        # generate random number between 0 and 1, and another between 0 and 5 (inclusive)
        demo_num = np.random.randint(0, 2)
        progress = np.random.randint(1, 4)
        action_str = ['grasp', 'place'][np.random.randint(0, 2)]

        # get imgs
        d = demos[demo_num]
        color_heightmap, valid_depth_heightmap = d.get_heightmaps(action_str,
                d.action_dict[progress][action_str + '_image_ind'], use_hist=True)

        # get action info

        action_vec = d.action_dict[progress][ACTION_TO_ID[action_str]]

        # convert rotation angle to index
        best_rot_ind = np.around((np.rad2deg(action_vec[-2]) % 360) * 16 / 360).astype(int)

        # convert robot coordinates to pixel
        workspace_pixel_offset = workspace_limits[:2, 0] * -1 * 1000

        # need to index with (y, x) downstream, so swap order in best_pix_ind
        best_action_xy = ((workspace_pixel_offset + 1000 * action_vec[:2]) / 2).astype(int)
        best_pix_ind = [best_rot_ind, best_action_xy[1], best_action_xy[0]]

        # get reward
        reward = trial_rewards[progress + ACTION_TO_ID[action_str] - 1]

        # training step
        loss = trainer.backprop(color_heightmap, valid_depth_heightmap, action_str,
                best_pix_ind, reward, return_loss=True, silent=True)
        losses.append(loss.detach().cpu().data.numpy())

        # checkpoint
        if (i + 1) % 25 == 0:
            models[i] = trainer.model.state_dict()

    # get model with lowest + most stable loss
    min_loss = np.max(losses)
    best_model_ind = None
    for i in range(24, len(losses), 25):
        # avg loss across 5 consecutive steps
        avg_loss = np.mean(losses[i-2:min(i+3, len(losses))])
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_model_ind = i

    # create filenames and save best model
    if 'row' in args.base_model:
        base_name = 'row'
    elif 'unstack' in args.base_model:
        base_name = 'unstack'
    elif 'stack' in args.base_model:
        base_name = 'stack'
    else:
        base_name = 'vertical_square'

    model_name = '_'.join(['base', base_name, 'finetune', args.task_type])

    if args.out_dir is not None:
        print("Saving model at", best_model_ind + 1, "iterations with loss of", str(min_loss) + "...")
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        torch.save(models[best_model_ind], os.path.join(args.out_dir, model_name + '.pth'))

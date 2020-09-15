import numpy as np
import os
import argparse

# TODO(adit98) refactor to use ACTION_TO_IND from utils.py
# TODO(adit98) rename im_action.log and im_action_embed.log to be hyphenated

# 1) pick out successful grasp 'frames' (executed-action.log)
# 2) pick out corresponding embedding frame (im-action.log)
# 3) find nearest neighbor (euclidean distance) for imitation action embedding (im-action-embed -> executed-action-embed)
# 4) report % of matches (nearest neighbor matches executed action) - to find executed action (executed-action-embed -> executed-action)
# 5) report avg pixel distance (euclidean) between nearest neighbor and executed action

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--log_home', required=True, type=str, help='format is logs/EXPERIMENT_DIR')
args = parser.parse_args()

# load action success logs
grasp_successes = np.loadtxt(os.path.join(args.log_home, 'transitions', 'grasp-success.log.txt'))
place_successes = np.loadtxt(os.path.join(args.log_home, 'transitions', 'place-success.log.txt'))
action_success_inds = np.where(np.logical_or(grasp_successes, place_successes))[0]

# trim array length in case of premature exit
executed_actions = np.loadtxt(os.path.join(args.log_home, 'transitions', 'executed-action.log.txt'))[:grasp_successes.shape[0]]
im_actions = np.loadtxt(os.path.join(args.log_home, 'transitions', 'im_action.log.txt'))[:grasp_successes.shape[0]]

# load imitation embeddings and executed action embeddings
imitation_embeddings = np.load(os.path.join(args.log_home, 'transitions', 'im_action_embed.log.txt.npz'), allow_pickle=True)['arr_0']
executed_action_embeddings = np.load(os.path.join(args.log_home, 'transitions', 'executed-action-embed.log.txt.npz'), allow_pickle=True)['arr_0']

# store array to hold indices in the executed action embedding map that correspond to the imitation embedding
imitation_action_signal = []

# find nearest neighbor for each imitation embedding
for frame_ind, embedding in enumerate(executed_action_embeddings):
    l2_dist = np.sum(np.square(embedding - np.expand_dims(imitation_embeddings[frame_ind], axis=(0, 2, 3))), axis=1)
    match_ind = np.unravel_index(np.argmin(l2_dist), l2_dist.shape)
    imitation_action_signal.append(match_ind)

    # evaluate nearest neighbor distance for successful actions
    if frame_ind in action_success_inds:
        # TODO(adit98) calculate euclidean distance between match_ind and executed_action
        print('match_ind:', match_ind)
        print('executed_action ind:', executed_actions[frame_ind])

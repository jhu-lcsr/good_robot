import numpy as np
import os

# TODO(adit98) refactor to use ACTION_TO_IND from utils.py
# TODO(adit98) rename im_action.log and im_action_embed.log to be hyphenated

# 1) pick out successful grasp 'frames' (executed-action.log)
# 2) pick out corresponding embedding frame (im-action.log)
# 3) find nearest neighbor (euclidean distance) for imitation action embedding (im-action-embed -> executed-action-embed)
# 4) report % of matches (nearest neighbor matches executed action) - to find executed action (executed-action-embed -> executed-action)
# 5) report avg pixel distance (euclidean) between nearest neighbor and executed action

# TODO(adit98) define log_home with cmd line arg
log_home = 'logs/2020-09-14-19-11-43_Sim-Stack-Two-Step-Reward-Testing-Imitation'

# load action success logs
grasp_successes = np.loadtxt(os.path.join(log_home, 'transitions', 'grasp-success.log.txt'))
place_successes = np.loadtxt(os.path.join(log_home, 'transitions', 'place-success.log.txt'))

# trim array length in case of premature exit
executed_actions = np.loadtxt(os.path.join(log_home, 'transitions', 'executed-action.log.txt'))[:grasp_successes.shape[0]]
im_actions = np.loadtxt(os.path.join(log_home, 'transitions', 'im_action.log.txt'))[:grasp_successes.shape[0]]

# load imitation embeddings and executed action embeddings
imitation_embeddings = np.load(os.path.join(log_home, 'transitions', 'im_action_embed.log.txt.npy'), allow_pickle=True)
executed_action_embeddings = np.load(os.path.join(log_home, 'transitions', 'executed-action-embed.log.txt.npy'), allow_pickle=True)

print(imitation_embeddings.shape, executed_action_embeddings.shape)

# find nearest neighbor for each imitation embedding


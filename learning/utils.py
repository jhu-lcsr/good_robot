import os
import numpy as np
import torch
from imageio import imsave
from transformations import poses_m_to_px

import parameters.parameter_server as ps
from data_io.paths import get_results_dir

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_n_trainable_params(model):
    pp = 0
    for p in list(model.parameters()):
        if not p.requires_grad:
            continue
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def seq_to_batch_3d(image_sequence):
    return image_sequence.view([-1, image_sequence.size(2), image_sequence.size(3), image_sequence.size(4)]) if image_sequence is not None else None


def batch_to_seq_3d(image_batch, seq_len):
    return image_batch.view([-1, seq_len, image_batch.size(1), image_batch.size(2), image_batch.size(3)]) if image_batch is not None else None


def seq_to_batch_1d(vec_sequence):
    return vec_sequence.view([-1, vec_sequence.size(2)]) if vec_sequence is not None else None


def batch_to_seq_1d(vec_batch, seq_len):
    return vec_batch.view([-1, seq_len, vec_batch.size(1)]) if vec_batch is not None else None


def seq_to_batch_2d(mat_sequence):
    return mat_sequence.view([-1, mat_sequence.size(2), mat_sequence.size(3)]) if mat_sequence is not None else None


def batch_to_seq_2d(mat_batch, seq_len):
    return mat_batch.view([-1, seq_len, mat_batch.size(1), mat_batch.size(2)]) if mat_batch is not None else None


def layer_histogram_summaries(writer, name, layer, idx):
    writer.add_histogram(name + "/bias", layer.bias.data.cpu().numpy(), idx, bins="auto")
    writer.add_histogram(name + "/weight", layer.weight.data.cpu().numpy(), idx, bins="auto")


def draw_drone_poses(drone_poses):
    num_poses = len(drone_poses)
    pic = np.zeros([num_poses, 1, 128, 128])
    # TODO: Fix this call:
    poses_map = poses_m_to_px(drone_poses, 128, batch_dim=True)
    for i, pose in enumerate(poses_map):
        x = int(pose.position[0])
        y = int(pose.position[1])
        if x > 0 and y > 0 and x < 128 and y < 128:
            pic[i, 0, x, y] = 1.0

    return torch.from_numpy(pic)

def get_viz_dir_for_rollout():
    run_name = ps.get_current_run_name()
    import rollout.run_metadata as md
    instr_idx = md.CUSTOM_INSTR_NO
    env_id = md.ENV_ID
    seg_idx = md.SEG_IDX
    real_drone = md.REAL_DRONE
    vizdir = os.path.join(get_results_dir(run_name),
                f"viz_{'real' if real_drone else 'sim'}/{env_id}{'_' + str(seg_idx) if seg_idx is not None else ''}_{instr_idx}/")
    os.makedirs(vizdir, exist_ok=True)
    return vizdir


def save_tensor_as_img_during_rollout(tensor, name, prefix="", renorm_each_channel=False):
    tensor = tensor.data.cpu().numpy()
    tensor = tensor.transpose((1, 2, 0)).squeeze()

    if len(tensor.shape) > 2 and tensor.shape[2] == 2:
        extra_layer = np.zeros_like(tensor[:, :, 0:1])
        tensor = np.concatenate((tensor, extra_layer), axis=2)

    if renorm_each_channel and len(tensor.shape) > 2:
        for c in range(tensor.shape[2]):
            tensor[:, :, c] -= np.min(tensor[:, :, c])
            tensor[:, :, c] /= (np.max(tensor[:, :, c] + 1e-9))
    else:
        tensor -= np.min(tensor)
        tensor /= (np.max(tensor) + 1e-9)

    imsave(get_viz_dir_for_rollout() + name + prefix + ".png", tensor)
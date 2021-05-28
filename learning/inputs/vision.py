import numpy as np
from torch.autograd import Variable
import torch

from data_io.units import DEPTH_SCALE


def standardize_images(np_images, out_np=False):
    images_out = []
    for i, image in enumerate(np_images):
        images_out.append(standardize_image(image))
    if out_np:
        images_out = np.asarray(images_out)
    return images_out


def standardize_depth_images(np_images):
    images_out = []
    for i, image in enumerate(np_images):
        images_out.append(standardize_depth_image(image))
    return images_out


def standardize_depth_image(np_image):
    if np_image is None:
        return None

    channel = 0
    if len(np_image.shape) < 3:
        np_image = np.expand_dims(np_image, 2)

    if np_image.shape[2] > 3:
        channel = 3

    np_image = np.asarray(np_image).astype(float)
    np_image = np_image[:, :, channel:channel+1]
    np_image = np_image / DEPTH_SCALE
    np_image = np_image.transpose([2, 0, 1])
    return np_image


def standardize_image(np_image):
    if np_image is None:
        return None

    np_image = np.asarray(np_image).astype(np.float32)
    np_image = np_image[:, :, 0:3]
    np_image -= np.mean(np_image, axis=(0, 1, 2))
    np_image /= (np.std(np_image, axis=(0, 1, 2)) + 1e-9)
    np_image = np_image.transpose([2, 0, 1])
    return np_image


def standardize_2d_prob_dist(np_dist_stack):
    if np_dist_stack is None:
        return None

    np_image = np.asarray(np_dist_stack).astype(np.float32)
    assert np_image.min() >= 0, "Probability density must be positive"
    np_image /= (np_image.sum(axis=(0, 1), keepdims=True) + 1e-9)
    #print("normsum: ", np_image.sum(axis=(0, 1)))
    np_image = np_image.transpose([2, 0, 1])
    return np_image


def standardize_path_ground_truth(np_image):
    if np_image is None:
        return None

    np_image = np.asarray(np_image).astype(np.float32)
    np_image = np_image[:, :, 0:3]
    np_image -= np.min(np_image, axis=(0, 1, 2))
    np_image /= (np.max(np_image, axis=(0, 1, 2)) + 1e-9)
    np_image = np_image.transpose([2, 0, 1])
    return np_image


def viz_img(variable_or_tensor):
    if type(variable_or_tensor) is Variable:
        tensor = variable_or_tensor.data
    else:
        tensor = variable_or_tensor
    tensor = tensor.cpu()
    tensor = tensor - torch.min(tensor)
    tensor = tensor / (torch.max(tensor) + 1e-9)
    return tensor


def torch_to_np(variable_or_tensor):
    # If this is a torch Variable
    img = variable_or_tensor
    if type(img) is Variable:
        img = img.data.cpu()
    # If this is a torch tensor
    if hasattr(img, "cuda"):
        img = img.numpy()
        if len(img.shape) == 3:
            img = img.transpose((1, 2, 0))
        else:
            img = img.transpose((0, 2, 3, 1))

    # Normalize
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-9)

    return img
import numpy as np
import torch
from torch.autograd import Variable


def cuda_var(tensor, cuda=False, device=None):
    if cuda:
        tensor = tensor.contiguous().cuda(device=device, non_blocking=True)
    if type(tensor) == Variable:
        return tensor
    return Variable(tensor)


def empty_float_tensor(sizes, cuda=False, device=None):
    if cuda:
        tensor = torch.cuda.FloatTensor(*sizes)
    else:
        tensor = torch.FloatTensor(*sizes)
    tensor.fill_(0)
    return tensor


def np_to_tensor(np_tensor, insert_batch_dim=True, var=True, cuda=False):
    if isinstance(np_tensor, np.ndarray):
        tensor = torch.from_numpy(np_tensor).float()
    else: # Probably a lone float
        tensor = torch.FloatTensor([np_tensor])

    if insert_batch_dim:
        tensor = torch.unsqueeze(tensor, 0)
    if cuda:
        tensor = tensor.cuda()
    if var:
        tensor = Variable(tensor)
    return tensor
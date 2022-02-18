from collections import namedtuple

import torch
from torch.autograd import Variable
import parameters.parameter_server as P

Sample = namedtuple("Sample", ("instruction", "state", "action", "reward", "done", "metadata"))


def mask_tensors(tensors, mask):
    # If tensor and mask have the same number of elements
    result = []
    for tensor in tensors:
        if tensor.data.storage().size() == mask.data.storage().size():
            result.append(torch.masked_select(tensor, mask.view_as(tensor)))
        # Tensor has more dimensions and hence more elements
        elif tensor.dim() == 2:
            result.append(torch.masked_select(tensor, mask.expand_as(tensor)).view(-1, tensor.size(1)))
        elif tensor.dim() == 3:
            expand_mask = mask.unsqueeze(1)
            expand_mask = expand_mask.expand_as(tensor)
            result.append(torch.masked_select(tensor, expand_mask.expand_as(tensor)).view(-1, tensor.size(1), tensor.size(2)))
        elif tensor.dim() == 4:
            expand_mask = mask.unsqueeze(1).unsqueeze(1)
            expand_mask = expand_mask.expand_as(tensor)
            result.append(torch.masked_select(tensor, expand_mask.expand_as(tensor)).view(-1, tensor.size(1), tensor.size(2), tensor.size(3)))
        # If tensor has less dimensions than the mask, squeeze the mask
        elif tensor.dim() < mask.dim():
            result.append(torch.masked_select(tensor, mask.squeeze()))
        else:
            print(tensor.dim(), tensor.size())
            print ("Unsupported combination of inputs at mask_tensors")
    return tuple(result)


def instruction_sequence_batch_to_tensor(instructions, variable=True):
    traj_len = P.get_current_parameters()["Setup"]["trajectory_length"]

    bs = len(instructions)
    lenlist = []
    for b in range(bs):
        for instruction in instructions[b]:
            lenlist.append(len(instruction) if instruction is not None else 0)
    maxlen = max(lenlist)

    instructions_t = torch.LongTensor(bs, traj_len, maxlen + 1).fill_(0)
    lengths = torch.LongTensor(bs, traj_len).fill_(0)

    for b in range(bs):
        for i, instruction in enumerate(instructions[b]):
            if instruction is None:
                continue
            lengths[b][i] = len(instruction)
            instruction_padded = instruction + [0 for x in range(maxlen - len(instruction) + 1)]
            instructions_t[b][i] = torch.FloatTensor(instruction_padded)

    if instructions_t.dim() == 1:
        instructions_t = torch.unsqueeze(instructions_t, 0)

    if variable:
        instructions_v = Variable(instructions_t)
    else:
        instructions_v = instructions_t

    return instructions_v, lengths


def sequence_batch_to_tensor(instructions, variable=True):
    bs = len(instructions)
    lenlist = []
    for b in range(bs):
        for instruction in instructions[b]:
            lenlist.append(len(instruction) if instruction is not None else 0)
    maxlen = max(lenlist)

    instructions_t = torch.LongTensor(bs, maxlen + 1).fill_(0)
    lengths = torch.LongTensor(bs).fill_(0)

    for b in range(bs):
        for i, instruction in enumerate(instructions[b]):
            if instruction is None:
                continue
            lengths[b][i] = len(instruction)
            instruction_padded = instruction + [0 for x in range(maxlen - len(instruction) + 1)]
            instructions_t[b][i] = torch.FloatTensor(instruction_padded)

    if instructions_t.dim() == 1:
        instructions_t = torch.unsqueeze(instructions_t, 0)

    if variable:
        instructions_v = Variable(instructions_t)
    else:
        instructions_v = instructions_t

    return instructions_v, lengths


def sequence_list_to_tensor(instructions, variable=True):
    bs = len(instructions)
    lenlist = [len(instr) for instr in instructions]
    maxlen = max(lenlist)

    instructions_t = torch.LongTensor(bs, maxlen + 1).fill_(0)
    lengths = torch.LongTensor(bs).fill_(0)

    if variable:
        instructions_v = Variable(instructions_t)

    for i, instruction in enumerate(instructions):
        if instruction is None:
            continue
        lengths[i] = len(instruction)
        instruction_padded = list(instruction) + [0 for x in range(maxlen - len(instruction) + 1)]
        # Long tensors can't be created like this
        instructions_v[i] = torch.FloatTensor(instruction_padded)

    if instructions_v.dim() == 1:
        instructions_v = torch.unsqueeze(instructions_v, 0)

    return instructions_v, lengths


# TODO: The function above is the same as this, but returns lengths. This returns masks
def sequence_list_to_masked_tensor(instructions):
    """
    Takes a list of B sequences, each a list of integer tokens, possibly variable length, where the longest sequence
    is of length N.
    Produces a BxN sequence tensor containing the B sequences, each padded with zeroes if shorter than N.
    Also produces a BxN mask with one's for each element in the sequence tensor and zeroes for every padded element
    :param instructions: list of lists of integers, where each inner list is a tokenized instruction
    :return: tensor_instructions, tensor_mask
    """
    bs = len(instructions)
    lenlist = [len(instr) for instr in instructions]
    maxlen = max(lenlist)

    instructions_v = torch.LongTensor(bs, maxlen + 1).fill_(0)
    mask_v = torch.LongTensor(bs, maxlen + 1).fill_(0)

    for i, instruction in enumerate(instructions):
        if instruction is None:
            continue

        instruction_padded = list(instruction) + [0 for x in range(maxlen - len(instruction) + 1)]
        instruction_mask = [1 for i in range(0, lenlist[i])] + [0 for i in range(lenlist[i], maxlen + 1)]
        # Long tensors can't be created like this
        instructions_v[i] = torch.FloatTensor(instruction_padded)
        mask_v[i] = torch.FloatTensor(instruction_mask)

    if instructions_v.dim() == 1:
        instructions_v = instructions_v.unsqueeze(0)
        mask_v = mask_v.unsqueeze(0)

    return instructions_v, mask_v


def none_padded_seq_to_tensor(sequence, insert_batch_dim=False, cuda=False):
    bs = len(sequence)
    size = [bs, *sequence[0].shape]

    tensor_seq = torch.FloatTensor(*size).fill_(0)
    for i in range(bs):
        if sequence[i] is not None:
            tensor_seq[i] = torch.from_numpy(sequence[i]).float()
        else:
            break
    if insert_batch_dim:
        tensor_seq = torch.unsqueeze(tensor_seq, 0)

    if cuda:
        tensor_seq = tensor_seq.cuda()

    return tensor_seq


def pad_segment_with_nones(segment, max_seg_len):
    seg_rect = segment[:max_seg_len]
    while len(seg_rect) < max_seg_len:
        seg_rect.append({"instruction":None, "state":None, "ref_action": None, "reward": None, "done": None, "metadata": None})

    return seg_rect


def pad_batch_with_nones(seg_batch):
    max_seg_len = max([len(seg) for seg in seg_batch])
    # Clip segments at 80 to keep memory consumption reasonable
    if max_seg_len > 70:
        max_seg_len = 70
    seg_batch_rect = []
    for segment in seg_batch:
        seg_rect = segment[:max_seg_len]
        while len(seg_rect) < max_seg_len:
            seg_rect.append(Sample(None, None, None, None, None, None))
        seg_batch_rect.append(seg_rect)

    sample_batches = list(zip(*seg_batch_rect))
    return sample_batches, max_seg_len


def len_until_nones(sequence):
    l = 0
    for item in sequence:
        if item is None:
            break
        l += 1
    return l
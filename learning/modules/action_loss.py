import torch
import torch.nn as nn
from torch.autograd import Variable

from learning.inputs.sequence import mask_tensors
from learning.inputs.common import empty_float_tensor


class ActionLoss_Deprecated(torch.nn.Module):

    def __init__(self, run_name=""):
        super(ActionLoss_Deprecated, self).__init__()

        self.act_loss = nn.MSELoss(size_average=False, reduce=False)
        self.stoploss = nn.BCELoss(size_average=False, reduce=False)

    def forward(self, action_label, stop_label, action_pred, stop_pred, mask, reduce=False, metadata=None):
        fwd_pred = action_pred[:, 0]        # fwd velocity
        ang_pred = action_pred[:, 2]        # angular velocity
        fwd_label = action_label[:, 0]
        ang_label = action_label[:, 2]

        if mask is not None:
            mask = mask.unsqueeze(1).byte()
            (fwd_pred, ang_pred, stop_pred, fwd_label, ang_label, stop_label) = \
                mask_tensors((fwd_pred, ang_pred, stop_pred, fwd_label, ang_label, stop_label), mask)

        fwd_loss = self.act_loss(fwd_pred, fwd_label)
        ang_loss = self.act_loss(ang_pred, ang_label)
        stop_loss = self.stoploss(stop_pred, stop_label)

        if reduce:
            loss = 0.2 * fwd_loss + 0.6 * ang_loss + 0.2 * stop_loss
        else:
            loss = torch.cat([fwd_loss, ang_loss, stop_loss])

        nans = loss != loss
        if torch.sum(nans.long()).data.item() > 0:
            raise ValueError("Nan's encountered in loss calculation")

        return loss

    def write_summaries(self, prefix, idx, loss, avg_loss):
        full_prefix = self.model_name + "/" + prefix
        self.writer.add_scalar(full_prefix + "action_loss", avg_loss.data[0], idx)
        self.writer.add_scalar(full_prefix + "action_fwd_loss", loss.data[0], idx)
        self.writer.add_scalar(full_prefix + "action_ang_loss", loss.data[1], idx)
        self.writer.add_scalar(full_prefix + "action_stop_loss", loss.data[2], idx)


class ActionLoss(nn.Module):

    def __init__(self, run_name="", ang_weight=0.33, fwd_weight=0.33, stop_weight=0.33):
        super(ActionLoss, self).__init__()
        self.act_loss = nn.MSELoss(size_average=False, reduce=False)
        self.stoploss = nn.BCELoss(weight=torch.FloatTensor([0.8]), size_average=False)
        self.ang_weight = ang_weight
        self.fwd_weight = fwd_weight
        self.stop_weight = stop_weight

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.act_loss.cuda(device)
        self.stoploss.cuda(device)

    def forward(self, action_label, action_pred, mask=None, reduce=False, flags=None, batchreduce=True):
        fwd_pred = action_pred[:, 0]        # fwd velocity
        ang_pred = action_pred[:, 2]        # angular velocity
        stop_pred = action_pred[:, 3]
        fwd_label = action_label[:, 0]
        ang_label = action_label[:, 2]
        stop_label = action_label[:, 3]

        if mask is not None:
            mask = mask.unsqueeze(1).byte()
            (fwd_pred, ang_pred, stop_pred, fwd_label, ang_label, stop_label) = \
                mask_tensors((fwd_pred, ang_pred, stop_pred, fwd_label, ang_label, stop_label), mask)

        # Compute loss for each element in the batch
        fwd_loss = self.act_loss(fwd_pred, fwd_label)
        ang_loss = self.act_loss(ang_pred, ang_label)

        # Aggregate

        flagged_losses = {}
        """
        if flags is not None and None not in flags:
            batch_size = fwd_pred.size(0)
            seq_len = int(batch_size / len(flags))
            real_batch_size = int(batch_size / seq_len)
            for b in range(real_batch_size):
                for s in range(seq_len):
                    flag_loss = (0.2 * fwd_loss[b * seq_len + s] + 0.6 * ang_loss[b * seq_len + s]).data
                    flagged_losses[flags[b]] = flag_loss.cpu().numpy()[0]
        """

        if batchreduce:
            # Reduce the losses manually
            fwd_loss = torch.sum(fwd_loss)
            ang_loss = torch.sum(ang_loss)

            # Stop loss is already reduced, because PyTorch at the time of writing didn't have a reduce arg for it.
            stop_loss = self.stoploss(stop_pred, stop_label)
            loss = torch.cat([fwd_loss, ang_loss, stop_loss])

        else:
            stop_loss = torch.zeros_like(stop_pred)
            for i in range(len(stop_pred)):
                stop_loss[i:i+1] = self.stoploss(stop_pred[i:i+1], stop_label[i:i+1])
            loss = torch.stack([fwd_loss, ang_loss, stop_loss], dim=1)

        if reduce:
            loss = self.reduce_loss(loss)

        nans = loss != loss
        if torch.sum(nans.long()).data.item() > 0:
            print ("WARNING: Nan's encountered in loss calculation")
            print(loss)
            loss[nans] = 0
            return torch.zeros(list[loss.size()]).to(stop_pred.device)

        return loss, flagged_losses

    def batch_reduce_loss(self, loss):
        loss = torch.sum(loss, 0)
        return loss

    def reduce_loss(self, loss):
        total_loss = loss[0] * self.fwd_weight + loss[1] * self.ang_weight + loss[2] * self.stop_weight
        return total_loss

    def write_summaries(self, prefix, idx, loss, avg_loss):
        full_prefix = self.model_name + "/" + prefix
        self.writer.add_scalar(full_prefix + "action_loss", avg_loss.data[0], idx)
        self.writer.add_scalar(full_prefix + "action_fwd_loss", loss.data[0], idx)
        self.writer.add_scalar(full_prefix + "action_ang_loss", loss.data[1], idx)
        self.writer.add_scalar(full_prefix + "action_stop_loss", loss.data[2], idx)
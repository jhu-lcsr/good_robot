import torch
import numpy as np
import torch.nn as nn

from learning.modules.spatial_softmax_2d import SpatialSoftmax2d


class CrossEntropy2d(nn.Module):

    def __init__(self, run_name="", ang_weight=0.33, fwd_weight=0.33, stop_weight=0.33):
        super(CrossEntropy2d, self).__init__()
        self.softmax = SpatialSoftmax2d()
        self.logsoftmax = SpatialSoftmax2d(log=True)
        self.logsoftmax1d = nn.LogSoftmax(dim=1)

    def forward(self, pred, labels, oob_pred=None, oob_label=None):

        #x = - self.softmax(labels) * self.logsoftmax(pred)

        batch_size = pred.shape[0]
        channels = pred.shape[1]

        # Handle extra pixel that captures probability outside the masks
        if oob_pred is not None:
            assert oob_label is not None
            assert channels == 1, "When using oob pixels, only 1-channel probability maps are supported"
            # Concatenate oob_pred to pred and oob_label to label
            pred_flat = pred.view(batch_size, -1)
            pred_full = torch.cat([pred_flat, oob_pred[:, np.newaxis]], dim=1)

            labels_flat = labels.view(batch_size, -1)
            labels_full = torch.cat([labels_flat, oob_label[:, np.newaxis]], dim=1)
            x = -labels_full * self.logsoftmax1d(pred_full)
            # Sum over spatial dimensions:
            x = x.sum(1)
            # Average over channels and batches
            loss = torch.mean(x)

        # All probability mass is distributed over the masks
        else:
            x = - labels * self.logsoftmax(pred)

            # Sum over spatial dimensions:
            x = x.sum(2).sum(2)
            # Average over channels and batches
            loss = torch.mean(x)

        return loss
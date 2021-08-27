import torch.nn as nn

class PVN_Stage2_ValueHead(nn.Module):
    """
    Outputs a 4-D action, where
    """
    def __init__(self, h2=128):
        super(PVN_Stage2_ValueHead, self).__init__()
        self.linear = nn.Linear(h2, 1)

    def init_weights(self):
        pass

    def forward(self, features):
        x = self.linear(features)
        return x
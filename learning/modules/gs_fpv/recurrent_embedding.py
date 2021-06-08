import torch
from torch.nn import LSTM
import torch.nn as nn

from learning.inputs.common import cuda_var

class RecurrentEmbedding(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RecurrentEmbedding, self).__init__()

        self.lstm = LSTM(input_size, hidden_size, 1, True, False, 0, False)
        self.hidden_size = hidden_size

        self.last_h = None
        self.last_c = None

        self.hidden_size = hidden_size
        self.reset()
        self.dbg_t = None
        self.seq = 0

    def init_weights(self):
        pass

    def reset(self):
        self.last_h = torch.zeros([1, 1, self.hidden_size]).to(next(self.parameters()).device)
        self.last_c = torch.zeros([1, 1, self.hidden_size]).to(next(self.parameters()).device)

    def forward(self, inputs):
        outputs = self.lstm(inputs, (self.last_h, self.last_c))
        self.last_h = outputs[1][0]
        self.last_c = outputs[1][1]
        return outputs[0]
import torch 
import pdb 
import numpy as np
np.set_printoptions(precision=4, suppress = True, linewidth=400) 

class LSTMEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float, 
                 bidirectional:bool = True):
        super(LSTMEncoder, self).__init__() 
        self.input_dim = input_dim,
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout 
        self.bidirectional = bidirectional

        if bidirectional:
            self.output_size = 2 * self.hidden_dim
        else:
            self.output_size = self.hidden_dim
        # will be set later 
        self.device = torch.device("cpu") 

        self.lstm = torch.nn.LSTM(input_size  = input_dim,
                                  hidden_size = hidden_dim,
                                  num_layers  = num_layers, 
                                  bias        = True,
                                  batch_first = True,
                                  dropout     = dropout, 
                                  bidirectional = bidirectional)

    def set_device(self, device):
        self.device = device
        if "cuda" in str(device):
            self.lstm = self.lstm.cuda(device) 

    def forward(self, embedded_tokens, lengths):
        embedded_tokens = embedded_tokens.to(self.device) 
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded_tokens, 
                                                          lengths, 
                                                          batch_first=True, 
                                                          enforce_sorted=False) 
        packed_output, (hidden, cell) = self.lstm(embedded)
        output, lengths  = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        bsz, seq_len, hidden_dim = output.shape

        # concat them together  
        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, bsz, -1) 
            first = hidden[-1, 0, :, :].unsqueeze(1)
            last = hidden[-1, 1, :, :].unsqueeze(1) 
            concat = torch.cat([first, last], dim=1)
            
            # flatten out forward and backward 
            output = output.reshape(bsz, seq_len, -1) 
        else:
            hidden = hidden.view(self.num_layers, 1, bsz, -1) 
            concat = hidden 

        # flatten 
        concat = concat.reshape((bsz, -1))
        to_ret = {"sentence_encoding": concat,
                  "output": output} 
        return to_ret 
                                                              

class TransformerEncoder(torch.nn.Module): 
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim):
        super(TransformerEncoder, self).__init__() 
        #TODO (elias): Port miso code into here
        pass 

    def forward(self, embedded_tokens):
        pass 

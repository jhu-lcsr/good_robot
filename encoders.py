import torch 
import pdb 

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
        #print(f"embedded tokens {embedded_tokens[0, 0:10, 0:3]}") 
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded_tokens, lengths, 
                                                          batch_first=True, 
                                                          enforce_sorted=True) 
        embedded = embedded.to(self.device) 
        output, __ = self.lstm(embedded)
       
        output, lengths  = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        bsz, seq_len, hidden_dim = output.shape
        hidden_dim = int(hidden_dim / 2)
        output =  output.view((bsz, seq_len, 2, hidden_dim))
        # need indices of last non-pad token 
        lengths = lengths.long() 
        lengths = lengths-1
        batch_inds = torch.tensor([i for i in range(bsz)]).long() 
        # take first and last 
        first = output[:,0, 0,:].unsqueeze(1) 
        last  = output[batch_inds, lengths,1, :].unsqueeze(1) 
        # concat them together  
        if self.bidirectional:
            concat = torch.cat([first, last], dim=1)
            # flatten out forward and backward 
            output = output.reshape(bsz, seq_len, -1) 
        else:
            concat = last 

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

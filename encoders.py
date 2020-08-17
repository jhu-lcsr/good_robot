import torch 

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

        self.lstm = torch.nn.LSTM(input_size  = input_dim,
                                  hidden_size = hidden_dim,
                                  num_layers  = num_layers, 
                                  bias        = True,
                                  batch_first = True,
                                  dropout     = dropout, 
                                  bidirectional = bidirectional)

    def forward(self, embedded_tokens, lengths):
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded_tokens, lengths, 
                                                          batch_first=True, 
                                                          enforce_sorted=True) 
        output, __ = self.lstm(embedded)
        print(output.data.shape)
        return torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                                                              

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

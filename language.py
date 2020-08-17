from typing import Tuple
import torch


class SourceAttention(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        super(SourceAttention, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.q_proj = torch.nn.Linear(input_dim, output_dim) 
        self.k_proj = torch.nn.Linear(input_dim, output_dim) 
        self.v_proj = torch.nn.Linear(input_dim, output_dim) 

    def forward(self, q, k, v):
        # [batch, seq_len, output_dim]
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v) 
        # [batch, seq_len, seq_len]
        weights = torch.bmm(q, k.permute(0,2,1))
        # [batch, seq_len, output_dim] 
        output  = torch.bmm(weights, v) 
        return output 

class DeconvolutionalNetwork(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_shape: Tuple[int]):
        super(DeconvolutionalNetwork, self).__init__() 
        # TODO (elias): deal with bidirectional LSTM 
        self.input_dim = input_dim
        self.output_shape = output_shape

        #self.source_attn = SourceAttention(input_dim, attn_dim) 
        self.conv1 = torch.nn.ConvTranspose2d(128, 32, 16, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(32, 64, 17, padding=0)
        #self.up2   = torch.nn.Upsample((8, 128, 128))
        self.conv3 = torch.nn.ConvTranspose2d(64, 128, 33, padding=0)
        self.activation = torch.nn.ReLU() 
        # per-pixel output 
        #self.output= torch.nn.Linear()

    def forward(self, encoded):
        # encoded: [batch, seq_len, input_dim]
        # for now just take last one
        # [batch, 1, 1, input_dim]
        # one channel 
        print(encoded.data.shape)
        bsz, seq_len, input_dim = encoded.data.shape
        encoded = encoded[:,-1,:].unsqueeze(1).unsqueeze(2)
        # reshape [bsz, 4, 4, input_dim/8]
        encoded = encoded.reshape((bsz, -1, 1, 1))
        print(f"encoded before {encoded.shape}") 
        encoded = self.activation(self.conv1(encoded))
        print(f"conv 1: {encoded.shape}") 

        encoded = self.activation(self.conv2(encoded))
        print(f"conv 2: {encoded.shape}") 

        encoded = self.activation(self.conv3(encoded))
        print(f"conv 3: {encoded.shape}") 

        print(f"target shape: {self.output_shape}") 
        sys.exit() 
        # output: [batch, width, height] 
        output  = self.output(encoded)
        output  = output.reshape((bsz, self.output_shape[0], self.output_shape[1]))
        return output 


class LanguageEncoder(torch.nn.Module):
    """
    Handle language instructions as an API call to an encoder
    that tokenizes, embed tokens, and runs a selected encoder 
    over it, returning an output specified by the model.
    """
    def __init__(self,
                 embedder: torch.nn.Module,
                 encoder: torch.nn.Module,
                 output_type: str):
        """
        embedder: a choice of 
        encoder: a choice of LSTM or Transformer 
        output_type: choices are object mask, dense vector, 
        """
        super(LanguageEncoder, self).__init__() 

        self.embedder = embedder
        self.encoder = encoder
        self.output_type = output_type

        if self.output_type == "mask": 
            self.output_module = DeconvolutionalNetwork(self.encoder.hidden_dim, 64) 
        else:
            raise NotImplementedError(f"No output module for choice {output_type}") 


    def forward(self,
                language_batch: str) -> torch.Tensor: 
        embedded = []
        # TODO (elias): deal with padding 
        lengths = []
        for sent in language_batch:
            embedded_tokens = self.embedder(sent).unsqueeze(0)
            lengths.append(embedded_tokens.shape[1])
            embedded.append(embedded_tokens)

        max_seq = max(lengths) 

        for i, sent in enumerate(embedded):
            diff = [self.embedded.pad_token.unsqueeze(0) for i in range(max_seq - sent.shape[1])]
            if len(diff) > 0:
                diff = torch.cat(diff, dim = 0)
            else:
                continue
            sent = torch.cat([sent, diff], dim = 1)
            embedded[i] = sent 

        embedded = torch.cat(embedded, dim=1)
        lengths = torch.Tensor(lengths) 
        print(lengths) 
        print(embedded.shape)  
        # [bsz, len, hidden_dim]
        encoded_tokens, __ = self.encoder(embedded, lengths)
        output = self.output_module(encoded_tokens)
        return output 
        




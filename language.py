import torch

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
        super(self, LanguageEncoder).__init__() 

        self.embedder = embedder
        self.encoder = encoder
        self.output_type = output_type

        self.output_module = self.choose_output_module() 

    def choose_output_module(self) -> torch.nn.Module: 
        """
        choose an output transformation based on output module
        """
        if self.output_type == "mask": 
            output_module = DeconvolutionalNetwork(self.encoder.output_dim) 
        else:
            raise NotImplementedError(f"No output module for choice {self.output_type}") 
        return output_module 

    def forward(self,
                language_tokens: str) -> torch.Tensor: 
        embedded_tokens = self.embedder(language_tokens)
        encoded_tokens  = self.encoder(embedded_tokens)
        output = self.output_module(encoded_tokens)
        return output 
        




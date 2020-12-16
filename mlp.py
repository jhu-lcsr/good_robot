import torch


class MLP(torch.nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 dropout: float = 0.20
                ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        dropout_layer = torch.nn.Dropout(dropout) 

        layer0 = torch.nn.Linear(input_dim, hidden_dim)
        self.layers = [layer0, self.activation, dropout_layer]

        for i in range(self.num_layers-1):
            layer_i = torch.nn.Linear(hidden_dim, hidden_dim)
            self.layers += [layer_i, self.activation, dropout_layer]

        layer_n = torch.nn.Linear(hidden_dim, output_dim)
        self.layers.append(layer_n) 
        self.layers = torch.nn.ModuleList(self.layers)   

    def forward(self, inputs): 
        outputs = inputs

        for i, layer in enumerate(self.layers): 
            outputs = layer(outputs) 
        return outputs 
        
            
            


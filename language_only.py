import torch
from mlp import MLP 
import pdb 

class LanguageEncoder(torch.nn.Module):
    def __init__(self, 
                 embedder,
                 encoder,
                 device):
        super(LanguageEncoder, self).__init__() 

        self.lang_embedder = embedder
        self.lang_embedder.device = device
        self.lang_encoder = encoder
        self.lang_encoder.device = device

        self.mlp = MLP(input_dim = encoder.output_size,
                       hidden_dim = 64, 
                       output_dim = 21, 
                       num_layers = 3,
                       dropout = 0.20) 
        self.compute_block_dist = True 
        self.device = device

    def forward(self, data_batch):
        #pdb.set_trace() 

        lang_input = data_batch["command"]
        lang_length = data_batch["length"]
        # tensorize lengths 
        lengths = torch.tensor(lang_length).float() 
        lengths = lengths.to(self.device) 

        # embed language 
        lang_embedded = torch.cat([self.lang_embedder(lang_input[i]).unsqueeze(0) for i in range(len(lang_input))], 
                                    dim=0)
        # encode
        # USE CBOW FOR DEBUGGING 
        #mean_embedding = torch.sum(lang_embedded, dim = 1).repeat(1,2) 
        #lang_output = {"sentence_encoding": mean_embedding } 
        lang_output = self.lang_encoder(lang_embedded, lengths) 
        
        # get language output as sentence embedding 
        sent_encoding = lang_output["sentence_encoding"] 

        logits = self.mlp(sent_encoding)
        #print(torch.softmax(logits, dim = 1)[0])
        return {"pred_block_logits": logits} 


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_io.instructions import debug_untokenize_instruction

from learning.inputs.sequence import sequence_list_to_tensor
from learning.inputs.common import empty_float_tensor


# TODO: Parametrize
VOCAB_SIZE = 2080
class SentenceEmbeddingSimple(nn.Module):

    def __init__(self, word_embedding_size, embed_size, lstm_layers=1, dropout=0.5):
        super(SentenceEmbeddingSimple, self).__init__()
        self.lstm_size = embed_size
        self.lstm_layers = lstm_layers
        self.embedding = nn.Embedding(VOCAB_SIZE, word_embedding_size, sparse=False)
        self.lstm_txt = nn.LSTM(word_embedding_size, self.lstm_size, lstm_layers, dropout=0.0)
        self.dropout = nn.Dropout(dropout)

        self.last_output = None

    def init_weights(self):
        self.embedding.weight.data.normal_(0, 1)
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.lstm_txt.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def reset(self):
        self.last_output = None

    def get(self):
        return self.last_output

    def forward(self, word_ids, lengths=None):
        # TODO: Get rid of this and abstract in another layer
        device = self.embedding.weight.device
        if isinstance(word_ids, list) and lengths is None:
            word_ids, lengths = sequence_list_to_tensor([word_ids])
            word_ids = word_ids.to(device)
            lengths = lengths.to(device)

        batch_size = len(word_ids)
        sentence_embeddings = torch.zeros((batch_size, self.lstm_size)).to(device)

        last_instruction = None
        last_embedding = None

        for i in range(batch_size):
            length = int(lengths[i])
            if length == 0:
                #print("Empty caption")
                continue

            # If this instruction is same as the previous instruction, no reason to recompute the embedding for it
            this_instruction = word_ids[i:i+1, 0:length]
            #if this_instruction.dim() == 0:
            #    this_instr_list = [this_instruction.data.item()]
            #else:
            this_instr_list = list(this_instruction.data[0])

            if last_instruction is None or this_instr_list != last_instruction:
                embeddings_i = self.embedding(word_ids[i, 0:length]).unsqueeze(1)
                #embeddings_i = word_embeddings[i, 0:length].unsqueeze(1)
                #word_ids_i = word_ids[i, 0:length]
                #print(debug_untokenize_instruction(word_ids_i.data))

                h0 = torch.zeros((self.lstm_layers, 1, self.lstm_size)).to(device)
                c0 = torch.zeros((self.lstm_layers, 1, self.lstm_size)).to(device)
                outputs, states = self.lstm_txt(embeddings_i, (h0, c0))

                # Mean-reduce the 1st (sequence) dimension
                sentence_embedding = outputs[-1].squeeze()#torch.mean(outputs, 0)
                #sentence_embedding = self.dropout(sentence_embedding)
                last_embedding = sentence_embedding
                last_instruction = this_instr_list
                #print("computing")
            else:
                #print("skipping")
                sentence_embedding = last_embedding
            sentence_embeddings[i] = sentence_embedding.squeeze()

        # Assuming the batch is a sequence, keep the last embedding around
        # TODO: Revise this assumption
        self.last_output = sentence_embeddings[batch_size - 1 : batch_size]

        return sentence_embeddings

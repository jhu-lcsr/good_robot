import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from learning.inputs.sequence import sequence_list_to_tensor
from learning.inputs.common import empty_float_tensor
from data_io.paths import get_self_attention_path
import pickle
import matplotlib.pyplot as plt
import numpy as np

# TODO: Paramerize
VOCAB_SIZE = 2080
class SentenceEmbeddingSelfAttention(nn.Module):

    def __init__(self,
                 word_embedding_size,
                 lstm_size,
                 lstm_layers=1,
                 attention_heads=5,
                 run_name="", BiLSTM=True,
                 dropout = False):
        super(SentenceEmbeddingSelfAttention, self).__init__()
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.embedding = nn.Embedding(VOCAB_SIZE, word_embedding_size, sparse=False)
        self.BiLSTM = BiLSTM
        self.dropout = dropout
        self.factor = (2 if self.BiLSTM else 1)
        self.lstm_txt = nn.LSTM(word_embedding_size, self.lstm_size, self.lstm_layers, bidirectional=BiLSTM, dropout=0.5)
        self.Da = 25
        # TODO: Make sure the overall embedding is of the size requested
        self.num_attn_heads = attention_heads
        self.W_s1 = nn.Linear(self.factor * self.lstm_size, self.Da, bias = False)
        self.W_s2 = nn.Linear(self.Da, self.num_attn_heads, bias = False)
        self.init_weights()
        # self.W_s1 = nn.Parameter(torch.ones(self.Da, self.factor * self.lstm_size))
        # self.W_s2 = nn.Parameter(torch.ones(self.num_attn_heads, self.Da))

        # TODO: Use the proper way of loading this
        # self.idx2word = pickle.load(open(get_self_attention_path()+"idx2word.pickle", "rb"))
        self.idx2word = None

        self.n_epoch = 0
        self.n_batch = 0


    def init_weights(self):
        self.embedding.weight.data.normal_(0, 1)
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.lstm_txt.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def save_att_map(self, batch_num, epoch_num):
        self.n_batch = batch_num
        self.n_epoch = epoch_num
        return

    def forward(self, word_ids, lengths=None):
        # TODO: Get rid of this and abstract in another layer

        if isinstance(word_ids, list) and lengths is None:
            word_ids, lengths = sequence_list_to_tensor([word_ids])
            if self.is_cuda:
                word_ids = word_ids.cuda() #size: [2, 500] [batch size, max intruction len]
                lengths = lengths.cuda() #instruction length
        word_embeddings = self.embedding(word_ids) #size: [2, 500, 20] embedding size: 20
        batch_size = word_embeddings.size(0) # size:2
        sentence_embeddings = Variable(empty_float_tensor((batch_size, self.lstm_size*self.factor*self.num_attn_heads), self.is_cuda, self.cuda_device)) #size [2,80]

        penal = 0

        for i in range(batch_size):
            length = int(lengths[i])
            if length == 0:
                print("Empty caption")
                continue
            embeddings_i = word_embeddings[i, 0:length].unsqueeze(1) # size: [instruction length, 1, 20]
            h0 = Variable(empty_float_tensor((self.lstm_layers*self.factor, 1, self.lstm_size), self.is_cuda)) #size: [2, 1, 40]
            c0 = Variable(empty_float_tensor((self.lstm_layers*self.factor, 1, self.lstm_size), self.is_cuda)) #size: [2, 1, 40]
            outputs, states = self.lstm_txt(embeddings_i, (h0, c0)) #output size: [intr_len, 1, 80]  #2 states: forward and backwward.  size: [2, 1, 40]
            H = outputs.squeeze(dim=1) #size: [instr_len, 80]
            hidden, cell = (states[0].squeeze(dim=1), states[1].squeeze(dim=1)) #size: 2x[2,40]

            #self-attention
            s1 = self.W_s1(H)
            s2 = self.W_s2(F.tanh(s1))
            A = F.softmax(s2.t(), dim=1)
            M = torch.mm(A, H)

            AAt = torch.mm(A, A.t())
            for j in range(self.num_attn_heads):
                AAt[j, j] = 0
            p = torch.norm(AAt, 2)
            penal += p*p


        penal /= batch_size
        # Mean-reduce the 1st (sequence) dimension
        #sentence_embedding = torch.mean(M, 0) #size [80]
        sentence_embedding = M.view(-1)
        sentence_embeddings[i] = sentence_embedding.squeeze()

        if self.n_batch%2000 == 0 and self.idx2word is not None:
            str_id = word_ids[-1][:length].data.cpu().numpy()
            instr = [self.idx2word[str(i)] for i in str_id]
            Att = A.data.cpu().numpy()
            filepath = get_self_attention_path() + "sample_instructions/sample_intr-{}-{}.txt".format(self.n_epoch, self.n_batch)
            # with open(filepath, "w") as f:
            #     for w in zip(instr, Att[0], Att[1], Att[2], Att[3], Att[4]):
            #         f.write(str(w)+"\n")

            imgpath = get_self_attention_path() + "instruction_heatmap/intr_heatmap-{}-{}.png".format(self.n_epoch, self.n_batch)

            # plt.close()
            plt.figure(figsize=(len(instr) / 6, 1.8))
            plt.pcolor(Att)
            plt.xticks(np.linspace(0.5, len(instr)-0.5, len(instr)), instr, rotation=90, fontsize=10)
            plt.gcf().subplots_adjust(bottom=0.5)
            plt.savefig(imgpath)
            # plt.show()
            self.n_batch += 1

        return sentence_embeddings, penal
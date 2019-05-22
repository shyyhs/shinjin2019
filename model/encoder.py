import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model

class LSTMEncoder(nn.Module):
    def __init__(self, dict_size, args):
        super(LSTMEncoder, self).__init__()
        # init
        self.input_size = dict_size 
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.encoder_layer_num = args.encoder_layer_num
        self.dropout_rate = args.dropout_rate
        self.bidirectional = args.bidirectional
        self.device = args.device

        # embedding layer
        self.embedding = nn.Embedding(self.input_size, self.emb_size)

        # lstm layer
        self.lstm = nn.LSTM(input_size = self.emb_size, \
                hidden_size = self.hidden_size, \
                num_layers = self.encoder_layer_num, \
                bidirectional = self.bidirectional, \
                batch_first = True, \
                dropout = self.dropout_rate)

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        embedded = self.embedding(inputs)
        h0, c0 = self.init_hidden(batch_size)

        output, (hn, cn) = self.lstm(embedded, (h0, c0))

        hn = torch.transpose(hn,0,1)

        if (self.bidirectional == 1):
            hn = torch.cat((hn[:,-2,:],hn[:,-1,:]),dim=1)
        else:
            hn = hn[:,-1,:]
        # output: [batch, seqlen, hidden_size=512]
        # hn: [batch, layer_num * bidirectional, hidden_size=512]

        return output, hn

    def init_hidden(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        hidden_len = self.encoder_layer_num
        if (self.bidirectional): hidden_len*=2

        h0 = torch.zeros(hidden_len, \
                            batch_size, \
                            self.hidden_size, \
                            device=self.device)
        c0 = torch.zeros(hidden_len, \
                            batch_size, \
                            self.hidden_size, \
                            device=self.device)
        return (h0,c0)


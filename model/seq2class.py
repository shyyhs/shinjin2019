import torch
import torch.nn as nn
import random
import model

class BaseModel(nn.Module):

    def __init__(self, dict_size, args=None):
        super(BaseModel, self).__init__() 
        self.encoder = model.encoder.LSTMEncoder(dict_size, args)
        self.decoder = model.decoder.LinearDecoder(args)

    def forward(self, sample):
        output, hn= self.encoder(sample)
        res = self.decoder(hn)

        return res 

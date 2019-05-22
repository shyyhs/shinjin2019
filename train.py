import sys
import os
import time
import math
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ClassDataset
from model import *
from tensorboardX import SummaryWriter

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path,"r")))


def parser_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="where is the config file")
    args = parser.parse_args()
    config = read_config(args.config)
    args = vars(args)
    for key in args:
        if (key not in config):
            config[key] = args[key]
    return config

def train(model, train_dataset, dev_dataset, args):
    writer = SummaryWriter(args.log_path + args.writer_name)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    step = 0
    dev_step = 0
    for i in range(args.epoch):
        train_dataset.shuffle()
        for example in train_dataset.get_batch():
            step += 1
            source = example['source']
            target = example['target']
            predict = model.forward(source)

            optimizer.zero_grad()
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss",loss.item(), step)

        dev_dataset.shuffle()
        for example in dev_dataset.get_batch():
            dev_step += 1
            source = example['source']
            target = example['target']
            predict = model.forward(source)

            loss = criterion(predict, target)
            writer.add_scalar("dev loss",loss.item(), dev_step)
            del loss


        if (i % args.save_iter == 0):
            torch.save(model.state_dict(), args.log_path + "{}_{}.pt".format(args.writer_name,i))





args = parser_init()
print (args)


train_dataset = ClassDataset(args.train_path, args.vocab_path, args)
model = BaseModel(train_dataset.dict_size ,args)
model.to(args.device)
print ("model loaded: {}".format(model))

dev_dataset = ClassDataset(args.dev_path, args.vocab_path, args)
test_dataset = ClassDataset(args.test_path, args.vocab_path, args)
print ("dataset loaded")
train(model, train_dataset, dev_dataset, args)





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

def evaluate(model, test_dataset, args):
    TP, TN, FP, FN = 0, 0, 0, 0
    for example in test_dataset.get_batch():
        source = example['source']
        target = example['target'].item()
        predict_from_model = model.forward(source)
        predict = predict_from_model[0][1].item() > predict_from_model[0][0].item()       
        if (target == 1 and predict == 1):
            TP += 1
        if (target == 0 and predict == 0):
            TN += 1
        if (target == 1 and predict == 0):
            FN += 1
        if (target == 0 and predict == 1):
            FP += 1
    Precision = float(TP)/float(TP+FP)
    Recall = float(TP)/float(TP+FN)
    F1 = 2*(Precision*Recall)/(Precision+Recall)

    result_filename = os.path.join(args.result_path, os.path.basename(args.model_path) + ".txt")
    with open(result_filename, "w") as f:
        f.write("TP: {}\nTN: {}\nFP: {}\nFN: {}\nPrecision: {}\nRecall: {}\nF1: {}\n".format(TP, TN, FP, FN, Precision, Recall, F1))
    print("TP: {}\nTN: {}\nFP: {}\nFN: {}\nPrecision: {}\nRecall: {}\nF1: {}\n".format(TP, TN, FP, FN, Precision, Recall, F1))


args = parser_init()
args.batch_size = 1
print (args)

test_dataset = ClassDataset(args.test_path, args.vocab_path, args)
print ("dataset loaded")
model = BaseModel(test_dataset.dict_size ,args)
model.to(args.device)
model.load_state_dict(torch.load(args.model_path))
print ("model loaded: {}".format(model))
evaluate(model, test_dataset, args)



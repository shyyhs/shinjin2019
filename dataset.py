import os
import sys
import time
import random
import codecs

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle # shuffle source and target together
from torch.utils.data import Dataset, DataLoader

class ClassDataset(Dataset):
    """ dataset for classification """
    def __init__(self, datafile, dictfile, args=None):
        """
        Args:
            datafile: the path of training data (train.txt) 
            dictfile: the path of dict (vocab.txt)
        """
        # 0. init
        self.datafile = datafile
        self.dictfile = dictfile

        self.source = [] # list to save the sentences
        self.target = [] # list to save the judgement (-1 1)
        self.dict_size = 0 # how many words in the dataset
        self.data_size = 0 # how many lines in the datafile

        self.w2i = {} # word to index
        self.i2w = {} # index to word

        self.batch_size = 1 
        if (hasattr(args, "batch_size") == True):
            self.batch_size = args.batch_size
        self.batch_pos = 0

        self.max_len = 0 # if the length of some sentence > max_len, trunc these sentences to sentence[-max_len:]
        if (hasattr(args, "max_len") == True):
            self.max_len = args.max_len
        self.trunced_line = 0

        # device = gpu(cuda) or cpu
        if (hasattr(args, "device") == False):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if (torch.cuda.is_available()):
                self.device = torch.device(args.device)
            else:
                self.device = torch.device("cpu")

        if (os.path.exists(self.datafile)==0):
            print ("data file not found!")
            raise ValueError

        if (os.path.exists(self.dictfile)==0):
            print ("dict file not found!")
            raise ValueError

        # 1. read the dictionary
        with open(self.dictfile, "r") as f:
            lines = f.readlines()
            self.dict_size = len(lines)
            for i, line in enumerate(lines):
                word = line.strip() 

                self.w2i[word] = i # these two lines are important ..
                self.i2w[i] = word

        # 2. read the datafile
        with open(self.datafile, "r") as f:
            lines = f.readlines()
            self.data_size = len(lines) 
            for i, line in enumerate(lines):
                target_class, sentence = line.strip().split('\t') # target_class: 1, -1. sentence: japanese sentence

                if (self.max_len > 0 and len(sentence) > self.max_len):
                    sentence = sentence[-self.max_len:] # trunc long sentences
                    self.trunced_line += 1

                target_class = int(target_class)
                if (target_class == -1): # classes (-1 1) -> (0 1), because -1 is difficult to use when doing classification
                    target_class = 0

                self.target.append(target_class)
                self.source.append(self._sentence2idx(sentence))


    def _tokenize(self, line):
        return line.strip('\n').strip().split()


    def _sentence2idx(self, sentence): # sentence -> list of index 
        idx = []
        sentence = self._tokenize(sentence)
        for word in sentence:
            word_id = self.w2i.get(word)
            if (word_id == None): word_id = self.w2i.get('<unk>') # when meets unk
            idx.append(word_id)
        return idx 


    def _idx2sentence(self, idx): # list of index -> sentence
        sentence = []
        for i in idx:
            word = self.i2w[i]
            sentence.append(word)
        return sentence


    def __len__(self): 
        return self.data_size


    def __getitem__(self, i): # not in use
        sample = {'source': self.source[i],
                'target': self.target[i]}
        return sample


    def shuffle(self): # see sklearn.utils.shuffle
        self.batch_pos = 0
        self.source, self.target = shuffle(self.source, self.target)


    def get_batch(self):
        while (self.batch_pos + self.batch_size < self.data_size):
            batch_source = self.source[self.batch_pos: self.batch_pos + self.batch_size]
            batch_target = self.target[self.batch_pos: self.batch_pos + self.batch_size]
            self.batch_pos += self.batch_size
            s_maxlen = max([len(s) for s in batch_source])
            for i,s in enumerate(batch_source):
                s = s + [self.w2i['<pad>']] * (s_maxlen - len(s))
                batch_source[i] = s

            batch_source = torch.tensor(batch_source, \
                                        dtype = torch.long, \
                                        device = self.device)
            batch_target = torch.tensor(batch_target, \
                                        dtype = torch.long, \
                                        device = self.device)

            sample = {"source": batch_source,  \
                    "target": batch_target}

            yield sample # learn the usage of yield 


if (__name__ == "__main__"): # belows are all test code
    test = ClassDataset("/home/song/git/shinjin2019/acp-2.0/train.txt", "/home/song/git/shinjin2019/acp-2.0/vocab_50000.txt")
    print (test.__len__)
    word_num = 0
    unk_num = 0
    max_len = 0
    max_idx = 0
    avg_len = 0

    for i in range(test.__len__()):
        sample = test.__getitem__(i)
        sentence = sample['source']
        if (len(sentence)>max_len):
            max_len = len(sentence)
            max_idx = i
        avg_len += len(sentence)
        for word in sentence:
            if (word == test.w2i['<unk>']):
                unk_num += 1
            word_num += 1
        if (i%100000 == 0):
            print (sample)

    for i in range(3):
        test.shuffle()
        print(i)
        now = 0
        for sample in test.get_batch():
            now+=1
            if (now%10000==0):
                print (now)
            del sample


    print ("total word number: {}, total unknown word: {}, percentage: {}".format(word_num, unk_num, float(unk_num)/word_num))
    print ("max_len: {} max_len: {} avg_len: {}".format(max_len, max_idx,  float(avg_len)/test.__len__()))
    print ("max_len: {} trunced_line: {}".format(test.max_len, test.trunced_line))


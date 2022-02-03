import csv
import os
import torch
import random 
import pickle
import numpy as np
import pandas as pd
# from bpe import Encoder

class Vocab():

    # fit BPE
    # tokenize and encode
    # save , load

    def __init__(self, models_dir, data_path=None, train=None, test=None, dev=None):
        super().__init__()
        self.MODEL_DIR = models_dir
        self.TRAIN_DATA_PATH = data_path
        if train:
            self.load_train_data(self.TRAIN_DATA_PATH)
        if test:
            self.load_test_data(self.TRAIN_DATA_PATH)
        if dev:
            self.load_dev_data(self.TRAIN_DATA_PATH)

        self.encoder = Encoder(silent=True)
    
    def fit_data(self, path:str):
        with open(path) as infile:
          lines = list(map(str.strip, infile))[1:]
        new_lines = [cmt.split(",")[0] for cmt in lines]
        self.encoder.fit(new_lines)
    
    def tokenize_one(self, example):
        return (self.encoder.tokenize(example))
  
    def encodeing_one(self, example, tokenize=True, bpe=False):
        return next(self.encoder.transform([example]))

    def load_train_data(self, path: str):
        with open(path + 'train.json', "rb") as f_train:
          self.train_data = pickle.load(f_train)
        f_train.close()

    def load_dev_data(self, path: str):
        with open(path + 'dev.json', "rb") as f_dev:
          self.dev_data = pickle.load(f_dev)
        f_dev.close()

    def load_test_data(self, path: str):
        with open(path + 'test.json', "rb") as f_test:
          self.test_data = pickle.load(f_test)
        f_test.close()

    def save(self, dir_path: str = None):
        dir_path = dir_path or self.MODEL_DIR
        with open(dir_path, 'wb') as handle:
          pickle.dump(self.encoder, handle)
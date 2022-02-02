import csv
import os
import torch
import random 
import numpy as np
import pandas as pd
from bpe import Encoder


class Vocab():
  
    # fit BPE
    # tokenize and encode
    # save , load

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(silent=True)
        self.list_poet = []
        self.list_poem_tkn = []
        self.list_poem_bpe = []

    def fit_bpe(self, path_src:str, path_dst:str):
        with open(path_src) as infile:
          lines = list(map(str.strip, infile))
        self.encoder.fit(lines)
        df = pd.read_csv(path_dst, sep='delimiter')
        
        for i in range(len(df)-1) :
          poem, poet = df.loc[i, 0].split(",")
          self.list_poem_tkn.append(self.encoder.tokenize(poem))
          self.list_poem_bpe.append(next(self.encoder.transform([poem])))
          self.list_poet.append(poet)

        self.list_poem_bpe = self.list_poem_bpe[1:]
        self.list_poem_tkn = self.list_poem_tkn[1:]
        self.list_poet = self.list_poet[1:]
    
    def fit_one_bpe(self, example, tokenize=True, bpe=False):
      if tokenize and bpe:
        return (self.encoder.tokenize(example),next(self.encoder.transform([example])))
      elif bpe:
        return next(self.encoder.transform([example]))
      elif tokenize:
        self.encoder.tokenize(example)

    def load(self, path: str):
        df = pd.read_csv(path, sep='delimiter')
        for i in range(len(df)-1) :
          poem = df.loc[i, 0].split(",")
          poem, poet = poem[0], poem[1]
          self.list_poem_tkn.append(self.encoder.tokenize(poem))
          self.list_poem_bpe.append(next(self.encoder.transform([poem])))
          self.list_poet.append(poet)

    def save(self, path: str):
        with  open(path, 'w') as myfile:
          wr = csv.writer(myfile)
          wr.writerow(["poem_tokenize", "poem_bpe", "poet"])
        for cmd in range(len(self.list_poem_tkn)):
          wr.writerow([self.list_poem_tkn[cmd], self.list_poem_bpe[cmd], self.list_poet[cmd]])
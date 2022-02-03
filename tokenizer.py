import csv
import os
from typing import List
import torch
import random
import pickle
import numpy as np
import pandas as pd
from bpe import Encoder

extra_special_tokens = [
    '__style1',
    '__style2',
    '__style3',
    '__style4',
    '__style5',
    '__style6',
    '__extra1', # -> Use as BOS, EOS, other special tokens
    '__extra2',
    '__extra3',
    '__extra4',
]

snp_config = {
    'vocab_size': 6_000,
    'pct_bpe': 0.1,
    'silent': True,
    'ngram_min': 2,
    'ngram_max': 8,
    'required_tokens': extra_special_tokens,
    'strict': False,
}

# Not optimal
poem_config = {
    'vocab_size': 10_000,
    'pct_bpe': 0.2,
    'silent': True,
    'ngram_min': 1,
    'ngram_max': 8,
    'required_tokens': extra_special_tokens,
    'strict': False,
}

#TODO: make load a class method to facilitate loading and saving different tokenizers
class Tokenizer():
    # Don't change these!    
    EOW = '__eow'
    SOW = '__sow'
    UNK = '__unk'
    PAD = '__pad'

    def __init__(self,
                 name: str, load: bool = True, type: str = 'snp',
                 models_dir: str = './models_dir/', data_file: str = None,
                 ) -> None:
        self.config = snp_config if type == 'snp' else poem_config
        self.name = name
        self.type = type
        self.MODELS_DIR = models_dir
        self.DATA_FILE = data_file
        self.encoder = Encoder(**self.config)

    def tokenize(self, sentences) -> List[List[str]]:
        '''
        Given a list of sentences, tokenizes each of them
        '''
        return [self.encoder.tokenize(s) for s in sentences]

    def transform(self, sentences: List[str]) -> List[List[int]]:
        '''
        Given a list of sentences, transforms each of them into list of integers (token_ids)
        '''
        return self.encoder.transform(sentences)

    def inv_transform(self, sentences: List[List[int]]) -> List[str]:
        '''
        Given a set of token_id sequences, transforms each of them back into a sentence
        '''
        return self.encoder.inverse_transform(sentences)

    def save(self, file_name: str = None) -> None:
        file_name = self.name if file_name is None else file_name
        self.encoder.save(os.path.join(self.MODELS_DIR, file_name))

    def load(self, file_name: str = None) -> None:
        file_name = self.name if file_name is None else file_name
        self.encoder = self.encoder.load(os.path.join(self.MODELS_DIR, file_name))

    def fit(self):
        assert self.DATA_FILE is not None, 'You have not specified training data for the tokenizer'
        corpus = []
        with open(self.DATA_FILE, encoding='utf-8') as f:
            for row in csv.reader(f):
                corpus.append(row[0])
        self.encoder.fit(corpus)

# t = Tokenizer(name='poem_tokenizer', load=False, type_='poem', models_dir='./models_dir/', data_file='./data/poems/train.csv')
# t = Tokenizer(name='snp_tokenizer', load=False, type_='snp', models_dir='./models_dir/', data_file='./data/snappfood/train.csv')
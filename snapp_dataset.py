import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from datasets import Dataset
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tokenizer import Tokenizer


def get_snapp_dataset(data_df, tokenizer):
    labels = data_df['label'].to_list()
    labels_ = []

    for data in labels:
        labels_.append(0 if data == "SAD" else 1)

    text_list = data_df['comment'].to_list()
    df = pd.DataFrame({'text': tokenizer.transform(
        data_df['comment'].to_list()), 'label': labels_})
    hg_dataset = Dataset(pa.Table.from_pandas(df))
    return hg_dataset


def create_snapp_dataset_from_path(path, filename, tokenizer):
    data = pd.read_csv(path + filename)
    return get_snapp_dataset(data, tokenizer)


def get_no_peek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def get_padding_mask(txt_batch):
    B = len(txt_batch)
    seq_lens = [len(d) for d in txt_batch]
    T = max(seq_lens)

    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    for i, length in enumerate(seq_lens):
        padding_mask[i, length:] = True
    return padding_mask


def data_collator_snapp(batch):
    '''
    :param batch: a batch of texts (already tokenized) and their respective labels
    :return: adjusted inputs for encoder and decoder, src_key_padding_mask, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask
    '''
    EOS_token_id = 5
    '''
    NOTE for ourselves: batch is a list of dicts. dict[i] is a dictionary with a "text" and a "label" field
    '''

    texts = [torch.tensor(d['text'] + [EOS_token_id],
                          dtype=torch.int64) for d in batch]
    labels = [d['label'] for d in batch]
    labels = torch.tensor(labels, dtype=torch.int64)

    src_key_padding_mask = get_padding_mask(texts)
    tgt_mask = get_no_peek_mask(src_key_padding_mask.shape[1])  # T
    # in any case, we can can use torch.roll(x, 1, 1) to shift right in the rows

    '''
    encoder_input, -> decoder_tgt
    decoder_labels
    [a,b,c,EOS,pad pad] -> [BOS, a,b,c,EOS,pad]
    '''

    text_batch_in = pad_sequence(
        texts, batch_first=True, padding_value=0)  # inputs for encoder

    return text_batch_in, labels, src_key_padding_mask, tgt_mask

# snp_tokenizer = Tokenizer(name='snp_tokenizer', load=True, type_='snp', models_dir='./models_dir/',
#                           data_file='./data/snappfood/train.csv')
# print(snp_tokenizer.encoder.word_vocab['__extra4'], snp_tokenizer.encoder.word_vocab[snp_tokenizer.PAD]) # 5, 0
# print(snp_tokenizer.encoder.word_vocab['__style1'], snp_tokenizer.encoder.word_vocab['__style2']) # 3, 2

# dev_dataset = create_snapp_dataset_from_path("data/snappfood/", "dev.csv", snp_tokenizer)

# valid_loader = DataLoader(dev_dataset, batch_size=4, shuffle=True, collate_fn=data_collator_snapp)

# total_step = 0


#     ## TODO: what do we want in the training loop?? we can change data_collator later for that (adjusting shifts and stuff)

# print(total_step)
# print(len(valid_loader))

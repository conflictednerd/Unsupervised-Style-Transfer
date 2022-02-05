import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
from tokenizer import Tokenizer
import torch
from einops import rearrange
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def get_snapp_dataset(data_df, tokenizer):
    labels = data_df['label'].to_list()
    labels_ = []

    for data in labels:
        labels_.append(0 if data == "SAD" else 1)

    text_list = data_df['comment'].to_list()
    print("$$$$$")
    print(text_list[:3])
    print("%%%%%")
    df = pd.DataFrame({'text': tokenizer.transform(data_df['comment'].to_list()), 'label': labels_})
    hg_dataset = Dataset(pa.Table.from_pandas(df))
    return hg_dataset


def create_snapp_dataset_from_path(path, filename, tokenizer):
    data = pd.read_csv(path + filename)
    print(data.head())
    return get_snapp_dataset(data, tokenizer)


def get_no_peek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def get_padding_mask(batch):
    B = len(batch)
    seq_lens = [len(d) for d in batch]
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

    ## src_key_padding_mask
    src_key_padding_mask = get_padding_mask(batch)
    tgt_mask = get_no_peek_mask(src_key_padding_mask.shape[1])
    memory_key_padding_mask = src_key_padding_mask.clone()
    tgt_key_padding_mask = src_key_padding_mask.clone()  ## why should we shift again? I feel like we should decide on the first token first before doing anything
    ## in any case, we can can use torch.roll(x, 1, 1) to shift right in the rows

    label_list, text_list, = [], []
    for dictionary in batch:
        _text = dictionary['text']
        _label = dictionary['label']
        label_list.append(_label)
        processed_text = torch.tensor((_text), dtype=torch.int64)
        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)

    text_list_input = pad_sequence(text_list, batch_first=True, padding_value=0)
    text_list_output = text_list_input.clone()

    return text_list_input, label_list, src_key_padding_mask, text_list_output, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask


snp_tokenizer = Tokenizer(name='snp_tokenizer', load=True, type_='snp', models_dir='./models_dir/',
                          data_file='./data/snappfood/train.csv')

dev_dataset = create_snapp_dataset_from_path("data/snappfood/", "dev.csv", snp_tokenizer)

valid_loader = DataLoader(dev_dataset, batch_size=4, shuffle=True, collate_fn=data_collator_snapp)

total_step = 0

for step, (src, label, src_key_padding_mask, tgt, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask) in enumerate(
        iter(valid_loader)):
    total_step += 1
    ## move these to 'cuda' here, not done in the collator
    
    ## TODO: what do we want in the training loop?? we can change data_collator later for that (adjusting shifts and stuff)

print(total_step)
print(len(valid_loader))

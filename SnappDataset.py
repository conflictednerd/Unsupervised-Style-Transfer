import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
from tokenizer import Tokenizer
import torch


def get_snapp_dataset(data_df, tokenizer):
    labels = data_df['label'].to_list()
    labels_ = []

    for data in labels:
        labels_.append(0 if data == "SAD" else 1)

    df = pd.DataFrame({'text': tokenizer.transform(data_df['comment'].to_list()), 'label': labels_})
    hg_dataset = Dataset(pa.Table.from_pandas(df))
    return hg_dataset


def create_snapp_dataset_from_path(path, filename, tokenizer):
    data = pd.read_csv(path + filename)
    print(data.head())
    return get_snapp_dataset(data, tokenizer)


# def data_collator_snap(batch):
#     '''
#
#     :param batch: a batch of texts (already tokenized) and their respective labels
#     :return: adjusted inputs for encoder and decoder, src_key_padding_mask, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask
#     '''
#     label_list, text_list, = [], []
#
#     for (_text, _label) in batch:
#         label_list.append(_label)
#         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
#         text_list.append(processed_text)
#
#     label_list = torch.tensor(label_list, dtype=torch.int64)
#
#     text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
#
#     return text_list.to(device), label_list.to(device),

snp_tokenizer = Tokenizer(name='snp_tokenizer', load=True, type_='snp', models_dir='./models_dir/',
                          data_file='./data/snappfood/train.csv')

dev_dataset = create_snapp_dataset_from_path("data/snappfood/", "dev.csv", snp_tokenizer)
print(dev_dataset[0])
print([dev_dataset[0]['text']])
print([snp_tokenizer.encoder.bpe_vocab[snp_tokenizer.encoder.EOW]])
print([snp_tokenizer.encoder.bpe_vocab[
           66]])  ## It's the first token of that comment, yet it can't find it in either of bpe or word vocabularies
print(snp_tokenizer.inv_transform(
    [dev_dataset[0]['text']]))  ## TODO: why does this not transform the tokens back into the original text?
print(dev_dataset)

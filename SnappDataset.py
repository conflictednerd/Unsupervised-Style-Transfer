import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
from tokenizer import Tokenizer


def get_snapp_dataset(data_df):
    labels = data_df['label'].to_list()
    labels_ = []
    snp_tokenizer = Tokenizer(name='snp_tokenizer', load=True, type_='snp', models_dir='./models_dir/',
                              data_file='./data/snappfood/train.csv')
    for data in labels:
        labels_.append(0 if data == "SAD" else 1)

    df = pd.DataFrame({'text': snp_tokenizer.transform(data_df['comment'].to_list()), 'label': labels_})
    hg_dataset = Dataset(pa.Table.from_pandas(df))
    return hg_dataset


def create_snapp_dataset_from_path(path, filename):
    data = pd.read_csv(path + filename)
    print(data.head())
    return get_snapp_dataset(data)


dev_dataset = create_snapp_dataset_from_path("data/snappfood/", "dev.csv")  ## TODO: fix error "KeyError: '__unk'"
print(dev_dataset)

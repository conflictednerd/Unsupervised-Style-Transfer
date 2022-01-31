import math
from typing import List

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # comment to learn positional embeddings
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] if batch_first = False
        """
        if self.batch_first:
            x = (x.permute(1, 0, 2) + self.pe[:x.size(1)]).permute(1, 0, 2)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(
        self, d_model: int, nhead: int,
        dim_feedforward: int, num_layers: int, vocab_size: int,
        batch_first: bool, device
    ) -> None:

        super().__init__()
        self.device = device
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, batch_first=batch_first)
        self.model = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=batch_first, device=device),
            num_layers=num_layers)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.2
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor, seq_lens: List[int]) -> torch.Tensor:
        assert x.shape[0] == len(seq_lens)  # = batch_size
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        src_key_padding_mask = self.get_padding_mask(
            B=x.shape[0], T=x.shape[1], seq_lens=seq_lens)
        x = self.model(src=x, src_key_padding_mask=src_key_padding_mask)
        return x

    # TODO: should be done in the dataset?
    def get_padding_mask(self, B: int, T: int, seq_lens: List[int]) -> torch.Tensor:
        '''
        Creates a BxT mask to identify padding tokens in each sequence

        Elements for which the mask is True, do not contribute in the self attention mechanism
        (its as if they don't exist when computing other elements embeddings).
        '''

        src_key_padding_mask = torch.zeros(B, T, dtype=torch.bool)
        for i, length in enumerate(seq_lens):
            src_key_padding_mask[i, length:] = True
        return src_key_padding_mask


# m = Encoder(256, 8, 1024, 4, 10000, True, 'cpu')
# print(sum(p.numel() for p in m.parameters() if p.requires_grad))
# # batch of 16 sequences, each with length 256
# x = torch.randint(0, 10000, (16, 256))
# seq_lens = [100]*16
# out = m(x, seq_lens)
# print(out.shape)

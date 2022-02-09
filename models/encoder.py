import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


'''
ATTENTION!:
we should use ONE optimizer for both the embedding layer AND the encoder
'''


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, batch_first) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, batch_first=batch_first)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.2
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self, d_model: int, nhead: int,
        dim_feedforward: int, num_layers: int, vocab_size: int,
        batch_first: bool, device
    ) -> None:

        super().__init__()
        self.device = device
        self.d_model = d_model
        self.model = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=0.3,
            activation=F.leaky_relu, #TODO: lambda can't be pickled
            dim_feedforward=dim_feedforward, batch_first=batch_first, device=device),
            num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        x = self.model(src=x, src_key_padding_mask=src_key_padding_mask)
        return x


# m = Encoder(256, 8, 512, 2, 6000, True, 'cpu')
# emb = EmbeddingLayer(6_000, 256, True)
# x = torch.tensor([4]*16).unsqueeze(-1)
# print(x.shape)
# print(emb(x).shape)
# print(sum(p.numel() for p in m.parameters() if p.requires_grad))
# print(sum(p.numel() for p in emb.parameters() if p.requires_grad))
# # batch of 16 sequences, each with length 256
# x = torch.randint(0, 6000, (16, 256))
# seq_lens = [100]*16
# out = m(emb(x), seq_lens)
# print(out.shape)

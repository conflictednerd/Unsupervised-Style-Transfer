import torch
import math
import torch.nn as nn
from encoder import PositionalEncoding
from encoder import Encoder  # to be commented


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, vocab_size, batch_first, device) -> None:
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, batch_first=batch_first)
        self.model = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=batch_first, device=device),
            num_layers=num_layers)

        self.init_weights()
        self.fc = nn.Linear(d_model, vocab_size)

    def init_weights(self) -> None:
        initrange = 0.2
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        '''
        (tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
        tgt: in our case, src that went into the input, but shifted to the right with the <BOS> token in the beginning
        tgt_mask: a list of masks, for auto-regression purposes
        memory: output of Encoder
        memory_mask: the mask for the memory sequence, which I think we can ignore since we have padding mask for it
        padding masks: t o ingnore paddings, but it should be true in the padding indices
        '''

        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        output = self.model(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask)

        return output


enc = Encoder(256, 8, 1024, 4, 10000, True, 'cpu')
print(sum(p.numel() for p in enc.parameters() if p.requires_grad))
x = torch.randint(0, 10000, (16, 128))
seq_lens = [100] * 16
out_enc = enc(x, seq_lens)

dec = Decoder(256, 8, 1024, 4, 10000, True, 'cpu')
print(sum(p.numel() for p in dec.parameters() if p.requires_grad))
out_dec = dec(x, out_enc)
print(out_dec.shape)

import torch
import torch.nn as nn

from encoder import EmbeddingLayer, Encoder  # to be commented


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, vocab_size, batch_first, device) -> None:
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.model = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=batch_first, device=device),
            num_layers=num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        '''
        (tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
        tgt: in our case, src that went into the input, but shifted to the right with the <BOS> token in the beginning
        tgt_mask: a list of masks, for auto-regression purposes (the causal mask to avoid letting the decoder look at the next target token)
        memory: output of Encoder
        memory_mask: the mask for the memory sequence, which I think we can ignore since we have padding mask for it
        padding masks: t o ingnore paddings, but it should be true in the padding indices

        for our purposes, memory_key_padding_mask and tgt_key_padding_mask are both the same and tgt_mask is a lower triangular causal mask
        '''

        output = self.model(
            tgt=tgt, memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask)

        return output


# enc = Encoder(256, 8, 512, 4, 6000, True, 'cpu')
# emb = EmbeddingLayer(6_000, 256, True)
# print(sum(p.numel() for p in enc.parameters() if p.requires_grad))

# x = torch.randint(0, 6000, (16, 256))
# seq_lens = [100]*16
# out_enc = enc(emb(x), seq_lens)

# dec = Decoder(256, 8, 512, 4, 6000, True, 'cpu')
# print(sum(p.numel() for p in dec.parameters() if p.requires_grad))
# y = torch.randint(0, 6000, (16, 1))
# out_dec = dec(emb(y), out_enc)
# print(out_dec.shape)

'''
<BOS> t_1 -> v_1

BxTx1 -> BxTxh (EMB)
To the end of x add EOS
shift to right, add BOS
add BOS to the beginning of x (org input) and remove <EOS> (BxT+1x1)
memory (BxTxh enc's output) tgt (BxTxh(BOS[label] <-> EOS in input seq)) label (input sequence)

'''

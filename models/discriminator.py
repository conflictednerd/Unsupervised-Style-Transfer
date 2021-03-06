from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add instance norm?

# Adapted from https://github.com/mberaha/style_text/blob/master/src/discriminator.py


class CNNDiscriminator(nn.Module):
    '''
        GAN discriminator is a TextCNN
    '''

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int],
                 hidden_size: int, num_classes: int, dropout: float = 0.3) -> None:
        '''
        Args:
        in_channels -- the input feature maps. Should be only one for text.
        out_channels -- the output feature maps a.k.a response maps
                        = number of filters/kernels
        kernel_sizes -- the lengths of the filters. Should be the number of
                        generators' hidden states sweeped at a time
                        by the different filters.
        hidden_size -- size of the hidden states of the generator (d_model)
        hidden_units = is the number of hidden units for the Linear layer
        '''
        super().__init__()

        self.dropoutLayer = nn.Dropout(p=dropout)
        # build parallel CNNs with different kernel sizes
        self.convs = nn.ModuleList([])
        for ks in kernel_sizes:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(ks, hidden_size),
                stride=1)
            self.convs.append(conv)
        self.linear1 = nn.Linear(out_channels*len(kernel_sizes), 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        x -- (batch_size(B), seq_len(T), hidden_size(d))
            = (1, seq_length (max_length for professor), hidden_size)
        '''
        x = x.unsqueeze(1)  # unsqueeze to add channel dimension
        x = [
            F.leaky_relu(conv(x)).squeeze(3)
            for conv in self.convs]  # x[i].shape = [B x out_channels x T-kernel_sizes[i]]
        # perform max pooling over the entire sequence
        x = [
            F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # x[i].shape = [B x out_channels]

        # x.shape = [B x (out_channels * len (kernel_sizes) )]
        x = torch.cat(x, 1)
        x = self.dropoutLayer(x)
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)

        return x


# model = CNNDiscriminator(in_channels=1, out_channels=4, kernel_sizes=[
#                          1, 2, 3, 4, 5, 6, 8, 10], hidden_size=256, num_classes=3)
# # for snappfood: kernel_sizes = [1,2,3,4,5,6,8,10] -> 40k
# # for poems: kernel_sizes = [1,2,3,4,5,6,8,10,16,32,64,128], out_channels=4 -> 300k
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# x = torch.rand(16, 512, 256)

# out = model(x)
# print(out.shape)

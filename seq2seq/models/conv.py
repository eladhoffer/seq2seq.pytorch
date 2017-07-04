import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import LayerNorm1d, MaskedConv1d, GatedConv1d
from .seq2seq_base import Seq2Seq

class StackedConv(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size=3,
                 num_layers=4, bias=True,
                 dropout=0, causal=True):
        super(StackedConv, self).__init__()
        self.convs = nn.ModuleList()
        size = input_size
        for l in range(num_layers):
            self.convs.append(GatedConv1d(size, hidden_size, 1, bias=bias,
                                          causal=False))
            self.convs.append(nn.BatchNorm1d(hidden_size))
            self.convs.append(MaskedConv1d(hidden_size, hidden_size,
                                           kernel_size, bias=bias,
                                           groups=hidden_size,
                                           causal=causal))
            self.convs.append(nn.BatchNorm1d(hidden_size))
            size = hidden_size

    def forward(self, x):
        res = None
        for conv in self.convs:
            x = conv(x)
            if res is not None:
                x = x + res
            res = x
        return x


class ConvEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, kernel_size=3,
                 num_layers=4, bias=True, dropout=0, causal=False):
        super(ConvEncoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.convs = StackedConv(hidden_size, hidden_size, kernel_size,
                                 num_layers, bias, causal=causal)

    def forward(self, inputs):
        x = self.embedder(inputs)
        x = x.transpose(1, 2)

        x = self.convs(x)
        return x


class ConvDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, kernel_size=3,
                 num_layers=4, bias=True, dropout=0, causal=True):
        super(ConvDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.convs = StackedConv(2 * hidden_size, hidden_size, kernel_size,
                                 num_layers, bias, causal=causal)

    def forward(self, inputs, state):
        x = self.embedder(inputs)
        x = x.transpose(1, 2)
        state = F.adaptive_avg_pool1d(state, x.size(2))
        x = torch.cat([x, state], 1)
        x = self.convs(x)
        x = x.transpose(1, 2)  # BxTxN
        x = x.contiguous().view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)  # BxTxN
        return x


class ConvSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256, kernel_size=3,
                 num_layers=4, bias=True, dropout=0):
        super(ConvSeq2Seq, self).__init__(batch_first=True)
        self.encoder = ConvEncoder(vocab_size, hidden_size=hidden_size, kernel_size=kernel_size,
                                   num_layers=num_layers, bias=bias, dropout=dropout, causal=False)
        self.decoder = ConvDecoder(vocab_size, hidden_size=hidden_size, kernel_size=kernel_size,
                                   num_layers=num_layers, bias=bias, dropout=dropout, causal=True)

    def forward(self, input_encoder, input_decoder):
        state = self.encoder(input_encoder)
        return self.decoder(input_decoder, state)

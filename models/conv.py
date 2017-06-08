import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import LayerNorm1d, MaskedConv1d, GatedConv1d


class RecurentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 mode='LSTM', dropout=0, bidirectional=False):
        super(RecurentEncoder, self).__init__()
        if mode not in ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']:
            raise ValueError( """An invalid option for `mode` was supplied,
                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        self.rnn = nn.RNNBase(mode, hidden_size, hidden_size,
                              num_layers, bias, batch_first,
                              dropout, bidirectional)
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.vocab_size = vocab_size

    def forward(self, inputs, hidden=None):
        x = self.embedder(inputs)
        x, hidden = self.rnn(x, hidden)
        return x, hidden


class RecurentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 mode='LSTM', dropout=0, bidirectional=False, tie_embedding=False):
        super(RecurentDecoder, self).__init__()
        if mode not in ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']:
            raise ValueError( """An invalid option for `mode` was supplied,
                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        self.rnn = nn.RNNBase(mode, hidden_size, hidden_size,
                              num_layers, bias, batch_first,
                              dropout, bidirectional)
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.embedder.weight = self.classifier.weight
        self.batch_first = batch_first
        self.vocab_size = vocab_size


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


class ConvSeq2Seq(nn.Module):

    def __init__(self, vocab_size, hidden_size=256, kernel_size=3,
                 num_layers=4, bias=True, dropout=0):
        super(ConvSeq2Seq, self).__init__()
        self.encoder = ConvEncoder(vocab_size, hidden_size=hidden_size, kernel_size=kernel_size,
                                   num_layers=num_layers, bias=bias, dropout=dropout, causal=False)
        self.decoder = ConvDecoder(vocab_size, hidden_size=hidden_size, kernel_size=kernel_size,
                                   num_layers=num_layers, bias=bias, dropout=dropout, causal=True)

    def forward(self, input_encoder, input_decoder):
        state = self.encoder(input_encoder)
        return self.decoder(input_decoder, state)

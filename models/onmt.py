import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import math
import pdb
from .seq2seq import Seq2Seq
from .attention import GlobalAttention


class Encoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 mode='LSTM', dropout=0, bidirectional=False):
        self.layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        input_size = hidden_size

        super(Encoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size,
                                     hidden_size,
                                     padding_idx=0)
        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            emb = pack(self.embedder(input[0]), lengths)
        else:
            emb = self.embedder(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return outputs, hidden_t


class StackedLSTM(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False, input_feed=False,
                 mode='LSTM', dropout=0, bidirectional=False, tie_embedding=False):
        self.layers = num_layers
        self.input_feed = input_feed
        input_size = hidden_size
        if self.input_feed:
            input_size += hidden_size

        super(Decoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size,
                                     hidden_size,
                                     padding_idx=0)
        self.rnn = StackedLSTM(num_layers, input_size,
                               hidden_size, dropout)
        self.attn = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

        self.hidden_size = hidden_size

    def forward(self, input, context):
        context, hidden = context
        emb = self.embedder(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = Variable(input.data.new(input.size(1),
                                          self.hidden_size).zero_(), requires_grad=False)
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        x = torch.stack(outputs)
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(input.size(0), input.size(1), -1)

        return x, (context, hidden)


class ONMTSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256,
                 num_layers=2, bias=True, dropout=0, tie_enc_dec_embedding=False):
        encoder = Encoder(vocab_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias, dropout=dropout)
        decoder = Decoder(vocab_size, hidden_size=hidden_size, tie_embedding=tie_enc_dec_embedding,
                          num_layers=num_layers, bias=bias, dropout=dropout)
        super(ONMTSeq2Seq, self).__init__(encoder, decoder)

        if tie_enc_dec_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

    def bridge(self, context):
        context, hidden = context
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        new_hidden = []
        for h in hidden:
            if self.encoder.num_directions == 2:
                new_h = h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                        .transpose(1, 2).contiguous() \
                        .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
            else:
                new_h = h
            new_hidden.append(new_h)
        return (context, tuple(new_hidden))

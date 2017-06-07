import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import math
import pdb


class Attention(nn.Module):
    """docstring for Attention."""

    def __init__(self, y_dim, x_dim, hidden_size):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(y_dim + x_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.SoftMax()

    def forward(self, y, xt):
        xt = xt.transpose(0, 1)
        B, T, x_dim = list(xt.size())
        y_dim = y.size(1)
        y_expanded = y.unsqueeze(1).expand(B, T, y_dim)
        inputs = torch.cat([y_expanded, xt], 2).view(B * T, x_dim + y_dim)
        hidden = self.linear1(inputs)
        output = self.linear2(hidden)  # (B*T)x1
        attention = self.softmax(output.view(B, T)).unsqueeze(1)
        weighted_xt = torch.bmm(attention, xt).squeeze(1)  # B x x_dim
        return weighted_xt, attention


class RecurentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0):
        super(RecurentEncoder, self).__init__()

        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                       bidirectional=True))
        self.rnn_layers.append(nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                       bidirectional=False))
        for n in range(num_layers - 2):
            self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                           bidirectional=False))
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.vocab_size = vocab_size

    def forward(self, inputs, hidden=None):
        x = self.embedder(inputs)
        x, _ = self.rnn_layers[0](x)
        x, _ = self.rnn_layers[1](x)
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x, _ = self.rnn_layers[i](x)
            x += residual
        return x


class RecurentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, context_size=128,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0):
        super(RecurentDecoder, self).__init__()

        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                       bidirectional=False))
        self.rnn_layers.append(nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                       bidirectional=False))
        for n in range(num_layers):
            self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                           bidirectional=False))
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.vocab_size = vocab_size

    def forward(self, inputs, hidden=None):
        x = self.embedder(inputs)
        x, _ = self.rnn_layers[0](x)
        x, _ = self.rnn_layers[1](x)
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x, _ = self.rnn_layers[i](x)
            x += residual
        return x


class StackedRecurrentAttention(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, context_size, dropout):
        super(StackedRecurrentAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.rnn_size = rnn_size
        self.attention = attention

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size + context_size, rnn_size))
            input_size = rnn_size

    def __forward_one(self, input, hidden, context):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        output = input
        for i, layer in enumerate(self.layers):
            weighted_context, attention = self.attention(output, context)
            output = torch.cat([output, weighted_context], 1)
            h_1_i, c_1_i = layer(output, (h_0[i], c_0[i]))
            output = h_1_i
            if i + 1 != self.num_layers:
                output = self.dropout(output)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return output, (weighted_context, attention), (h_1, c_1)

    def forward(self, input, context, hidden=None):
        packed_seq = isinstance(input, PackedSequence)

        if packed_seq:
            input, lengths = unpack(input)

        if hidden is None:
            zeros = input.data.new().resize_(
                self.num_layers, input.size(1), self.rnn_size).zero_()
            hidden = (Variable(zeros), Variable(zeros))

        outputs = []
        attentions = []
        contexts = []
        for emb_t in input.split(1):
            emb_t = emb_t.squeeze(0)
            output, (weighted_context, attention), hidden = self.__forward_one(
                emb_t, hidden, context)
            outputs.append(output)
            attentions.append(attention)
            contexts.append(weighted_context)

        outputs = torch.stack(outputs)
        attentions = torch.stack(attentions)
        contexts = torch.stack(contexts)

        if packed_seq:
            outputs = pack(outputs, lengths)
        return outputs, (contexts, attentions), hidden

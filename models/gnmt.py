import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import math
import pdb
from .seq2seq import Seq2Seq


class GNMT(nn.Module):

    def __init__(self, vocab_size, hidden_size=512,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0):
        super(GNMT, self).__init__()
        self.encoder = RecurrentEncoder(vocab_size, hidden_size,
                                        num_layers, bias, batch_first,
                                        dropout)
        self.encoder.cuda(1)
        self.decoder = RecurrentDecoder(vocab_size, hidden_size=hidden_size,
                                        context_size=hidden_size,
                                        attention_size=hidden_size,
                                        num_layers=num_layers,
                                        bias=bias,
                                        batch_first=batch_first,
                                        dropout=dropout)
        self.decoder.cuda(2)

    def forward(self, input_encoder, input_decoder, get_attention=False):
        input_encoder = input_encoder.cuda(1)
        context, _ = self.encoder(input_encoder)
        input_decoder = input_decoder.cuda(2)
        context = context.cuda(2)
        output, attention = self.decoder(input_decoder, context)
        output = output.cuda(0)
        attention = attention.cuda(0)

        if get_attention:
            return output, attention
        else:
            return output


class Attention(nn.Module):
    """docstring for Attention."""

    def __init__(self, y_dim, x_dim, hidden_size):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(y_dim + x_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax()

    def forward(self, y, xt):
        xt = xt.transpose(0, 1)
        B, T, x_dim = list(xt.size())
        y_dim = y.size(1)
        y_expanded = y.unsqueeze(1).expand(B, T, y_dim)
        inputs = torch.cat([y_expanded, xt], 2).view(B * T, x_dim + y_dim)
        hidden = self.linear1(inputs)
        output = self.linear2(hidden)  # (B*T)x1
        attention = self.softmax(output.view(B, T))
        weighted_xt = torch.bmm(attention.unsqueeze(1),
                                xt).squeeze(1)  # B x x_dim
        return weighted_xt, attention


class RecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0):
        super(RecurrentEncoder, self).__init__()

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
            x, hidden = self.rnn_layers[i](x)
            x = x + residual
        return x, hidden


class RecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, context_size=128, attention_size=128,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0):
        super(RecurrentDecoder, self).__init__()

        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(AttentionLSTMCell(
            hidden_size, context_size, hidden_size, attention_size, bias))
        self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                       bidirectional=False))
        for n in range(num_layers - 2):
            self.rnn_layers.append(nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                           bidirectional=False))
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, inputs, context, hidden=None):
        x = self.embedder(inputs)
        x, _, attentions = self.rnn_layers[0](x, context)
        context = x
        x, _, = self.rnn_layers[1](x)
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = torch.cat([x, context], 2)
            x, _ = self.rnn_layers[i](x)
            x = x + residual
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)
        return x, attentions


class AttentionLSTMCell(nn.Module):

    def __init__(self, input_size, context_size, hidden_size, attention_size=128, bias=True):
        super(AttentionLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size, bias=bias)
        self.attn = Attention(input_size, context_size, attention_size)

    def forward(self, inputs, context, hidden=None):
        if hidden is None:
            zeros = inputs.data.new().resize_(inputs.size(1), self.hidden_size).zero_()
            hidden = (Variable(zeros), Variable(zeros))
        outputs = []
        attentions = []
        for inputs_t in inputs.split(1):
            inputs_t = inputs_t.squeeze(0)
            hidden = self.cell(inputs_t, hidden)
            output, _ = hidden
            output, attention = self.attn(output, context)
            outputs += [output]
            attentions += [attention]

        outputs = torch.stack(outputs)
        attentions = torch.stack(attentions)
        return outputs, hidden, attentions

# #test:
# model = GNMT(32000)
# x1 = torch.autograd.Variable(torch.rand(16,32).long().fill_(2))
# x2 = torch.autograd.Variable(torch.rand(7,32).long().fill_(2))
# y = model(x1, x2, get_attention=True)

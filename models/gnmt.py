import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import math
import pdb
from .seq2seq import Seq2Seq


class GNMT(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0, gpu_assignment={'encoder': 1, 'decoder': 2}):
        super(GNMT, self).__init__()
        self.gpu_assignment = gpu_assignment
        self.encoder = RecurrentEncoder(vocab_size, hidden_size,
                                        num_layers, bias, batch_first,
                                        dropout)
        if self.gpu_assignment is not None and 'encoder' in self.gpu_assignment:
            self.encoder.cuda(self.gpu_assignment['encoder'])
        self.decoder = RecurrentDecoder(vocab_size, hidden_size=hidden_size,
                                        context_size=hidden_size,
                                        attention_size=hidden_size,
                                        num_layers=num_layers,
                                        bias=bias,
                                        batch_first=batch_first,
                                        dropout=dropout)
        if self.gpu_assignment is not None and 'decoder' in self.gpu_assignment:
            self.decoder.cuda(self.gpu_assignment['decoder'])

    def bridge(self, context):
        context, _ = context
        return (context, None)

    def forward(self, input_encoder, input_decoder, encoder_state=None):
        if self.gpu_assignment is not None and 'encoder' in self.gpu_assignment:
            input_encoder = input_encoder.cuda(self.gpu_assignment['encoder'])
            if encoder_state is not None:
                input_encoder = input_encoder.cuda(self.gpu_assignment['encoder'])

        context = self.encode(input_encoder, encoder_state)
        context = self.bridge(context)

        if self.gpu_assignment is not None and 'decoder' in self.gpu_assignment:
            input_decoder = input_decoder.cuda(self.gpu_assignment['decoder'])
            context = (context[0].cuda(2), context[1])

        output, hidden = self.decode(input_decoder, context)
        output = output.cuda(0)
        return output

# class Attention(nn.Module):
#     """docstring for Attention."""
#
#     def __init__(self, y_dim, x_dim, hidden_size):
#         super(Attention, self).__init__()
#         self.linear1 = nn.Linear(y_dim + x_dim, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, 1)
#         self.softmax = nn.Softmax()
#         self.tanh = nn.Tanh()
#
#     def forward(self, y, xt):
#         xt = xt.transpose(0, 1)
#         B, T, x_dim = list(xt.size())
#         y_dim = y.size(1)
#         y_expanded = y.unsqueeze(1).expand(B, T, y_dim)
#         inputs = torch.cat([y_expanded, xt], 2).view(B * T, x_dim + y_dim)
#         hidden = self.linear1(inputs)
#         hidden = self.tanh(hidden)
#         output = self.linear2(hidden)  # (B*T)x1
#         attention = self.softmax(output.view(B, T))
#         weighted_xt = torch.bmm(attention.unsqueeze(1),
#                                 xt).squeeze(1)  # B x x_dim
#         return weighted_xt, attention


class Attention(nn.Module):

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """
        input: batch x dim
        context:  sourceL x batch x dim
        """
        context = context.transpose(0, 1)
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn


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
        self.rnn_layers.append(AttentionLSTM(
            hidden_size, context_size, hidden_size, attention_size, bias))
        self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                       bidirectional=False))
        for n in range(num_layers - 2):
            self.rnn_layers.append(nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                           bidirectional=False))
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, inputs, context):
        enc_context, hidden = context
        x = self.embedder(inputs)
        x, _, attentions = self.rnn_layers[0](x, enc_context)
        dec_context = x
        x, _, = self.rnn_layers[1](x)
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = torch.cat([x, dec_context], 2)
            x, _ = self.rnn_layers[i](x)
            x = x + residual
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)
        return x, (enc_context, hidden)


class AttentionLSTM(nn.Module):

    def __init__(self, input_size, context_size, hidden_size, attention_size=128, bias=True):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size, bias=bias)
        # self.attn = Attention(input_size, context_size, attention_size)
        self.attn = Attention(hidden_size)

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

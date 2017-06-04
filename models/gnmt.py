import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import math
import pdb


#
# class SimpleLSTMCell(nn.Module):
#
#     def __init__(self, input_size, hidden_size, bias=True):
#         super(SimpleLSTMCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
#         self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, hidden):
#         hx, cx = hidden
#         gates = self.ih(input)
#         gates += self.hh(hx)
#
#         ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#
#         ingate = self.sigmoid(ingate)
#         forgetgate = self.sigmoid(forgetgate)
#         cellgate = self.tanh(cellgate)
#         outgate = self.sigmoid(outgate)
#
#         cy = (forgetgate * cx) + (ingate * cellgate)
#         hy = outgate * self.tanh(cy)
#
#         return hy, cy
#
# class StackedRecurrent(nn.Module):
#
#     def __init__(self, cell=, dropout):
#         super(StackedLSTM, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList()
#         self.rnn_size = rnn_size
#
#         for i in range(num_layers):
#             self.layers.append(nn.LSTMCell(input_size, rnn_size))
#             input_size = rnn_size
#
#     def __forward_step(self, input, hidden):
#         h_0, c_0 = hidden
#         h_1, c_1 = [], []
#         output = input
#         for i, layer in enumerate(self.layers):
#             h_1_i, c_1_i = layer(output, (h_0[i], c_0[i]))
#             output = h_1_i
#             if i + 1 != self.num_layers:
#                 output = self.dropout(output)
#             h_1 += [h_1_i]
#             c_1 += [c_1_i]
#
#         h_1 = torch.stack(h_1)
#         c_1 = torch.stack(c_1)
#
#         return output, (h_1, c_1)
#
#     def forward(self, input, hidden):
#         packed_seq = isinstance(input, PackedSequence)
#
#         if packed_seq:
#             input, lengths = unpack(input)
#
#         if hidden is None:
#             zeros = input.data.new().resize_(
#                 self.num_layers, input.size(1), self.rnn_size).zero_()
#             hidden = (Variable(zeros), Variable(zeros))
#
#         outputs = []
#         for emb_t in input.split(1):
#             emb_t = emb_t.squeeze(0)
#             output, hidden = self.__forward_step(emb_t, hidden)
#             outputs.append(output)
#
#         outputs = torch.stack(outputs)
#         if packed_seq:
#             outputs = pack(outputs, lengths)
#         return outputs, hidden

class RecurentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0):
        super(RecurentEncoder, self).__init__()
        if mode not in ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']:
            raise ValueError( """An invalid option for `mode` was supplied,
                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        rnn_layers = [
        nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True, dropout=dropout,
        bidirectional=True),
        nn.LSTM(2*hidden_size, hidden_size, num_layers=1, bias=True, dropout=dropout,
        bidirectional=False),
        nn.LSTM(hidden_size, hidden_size, num_layers=6, bias=True, dropout=dropout,
        bidirectional=False)
        )
        nn.LSTM(hidden_size, hidden_size,
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

    def forward(self, inputs, hidden=None):
        x = self.embedder(inputs)
        x, _ = self.rnn(x, hidden)
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)

        return x


#
#
# class StackedRecurrent(nn.Module):
#
#     def __init__(self, cell=, dropout):
#         super(StackedLSTM, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList()
#         self.rnn_size = rnn_size
#
#         for i in range(num_layers):
#             self.layers.append(nn.LSTMCell(input_size, rnn_size))
#             input_size = rnn_size
#
#     def __forward_one(self, input, hidden):
#         h_0, c_0 = hidden
#         h_1, c_1 = [], []
#         output = input
#         for i, layer in enumerate(self.layers):
#
#             h_1_i, c_1_i = layer(output, (h_0[i], c_0[i]))
#             output = h_1_i
#             if i + 1 != self.num_layers:
#                 output = self.dropout(output)
#             h_1 += [h_1_i]
#             c_1 += [c_1_i]
#
#         h_1 = torch.stack(h_1)
#         c_1 = torch.stack(c_1)
#
#         return output, (h_1, c_1)
#
#     def forward(self, input, hidden):
#         packed_seq = isinstance(input, PackedSequence)
#
#         if packed_seq:
#             input, lengths = unpack(input)
#
#         if hidden is None:
#             zeros = input.data.new().resize_(
#                 self.num_layers, input.size(1), self.rnn_size).zero_()
#             hidden = (Variable(zeros), Variable(zeros))
#
#         outputs = []
#         for emb_t in input.split(1):
#             emb_t = emb_t.squeeze(0)
#             output, hidden = self.__forward_one(emb_t, hidden)
#             outputs.append(output)
#
#         outputs = torch.stack(outputs)
#         if packed_seq:
#             outputs = pack(outputs, lengths)
#         return outputs, hidden
#
#
#
#
#
# class StackedAttentionLSTM(nn.Module):
#
#     def __init__(self, num_layers, input_size, rnn_size, dropout):
#         super(StackedAttentionLSTM, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList()
#
#         for i in range(num_layers):
#             self.layers.append(LSTMCell(input_size, rnn_size))
#             input_size = rnn_size
#
#     def forward(self, input, hidden):
#         h_0, c_0 = hidden
#         h_1, c_1 = [], []
#         for i, layer in enumerate(self.layers):
#             h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
#             input = h_1_i
#             if i + 1 != self.num_layers:
#                 input = self.dropout(input)
#             h_1 += [h_1_i]
#             c_1 += [c_1_i]
#
#         h_1 = torch.stack(h_1)
#         c_1 = torch.stack(c_1)
#
#         return input, (h_1, c_1)
#
#
# class Decoder(nn.Module):
#
#     def __init__(self, opt, dicts):
#         self.layers = opt.layers
#         self.input_feed = opt.input_feed
#         input_size = opt.word_vec_size
#         if self.input_feed:
#             input_size += opt.rnn_size
#
#         super(Decoder, self).__init__()
#         self.word_lut = nn.Embedding(dicts.size(),
#                                      opt.word_vec_size,
#                                      padding_idx=onmt.Constants.PAD)
#         self.rnn = StackedAttentionLSTM(
#             opt.layers, input_size, opt.rnn_size, opt.dropout)
#         self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
#         self.dropout = nn.Dropout(opt.dropout)
#
#         self.hidden_size = opt.rnn_size
#
#     def load_pretrained_vectors(self, opt):
#         if opt.pre_word_vecs_dec is not None:
#             pretrained = torch.load(opt.pre_word_vecs_dec)
#             self.word_lut.weight.data.copy_(pretrained)
#
#     def forward(self, input, hidden, context, init_output):
#         emb = self.word_lut(input)
#
#         # n.b. you can increase performance if you compute W_ih * x for all
#         # iterations in parallel, but that's only possible if
#         # self.input_feed=False
#         outputs = []
#         output = init_output
#         for emb_t in emb.split(1):
#             emb_t = emb_t.squeeze(0)
#             if self.input_feed:
#                 emb_t = torch.cat([emb_t, output], 1)
#
#             output, hidden = self.rnn(emb_t, hidden)
#             output, attn = self.attn(output, context.t())
#             output = self.dropout(output)
#             outputs += [output]
#
#         outputs = torch.stack(outputs)
#         return outputs, hidden, attn

    #
    #
    #     if rnn_type in ['LSTM', 'GRU']:
    #         self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
    #     else:
    #         try:
    #             nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
    #         except KeyError:
    #             raise ValueError( """An invalid option for `--model` was supplied,
    #                              options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
    #         self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
    #
    #         class RNNBase(Module):
    #
    # def __init__(self, mode, input_size, hidden_size,
    #              num_layers=1, bias=True, batch_first=False,
    #              dropout=0, bidirectional=False):

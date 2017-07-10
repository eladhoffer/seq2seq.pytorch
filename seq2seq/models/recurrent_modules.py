# Partially adapted from https://github.com/OpenNMT/OpenNMT-py

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from .seq2seq_base import Seq2Seq
from .attention import AttentionLayer
from seq2seq.tools.config import PAD




class StackedRecurrentCells(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout=0, bias=True, batch_first=False, rnn_cell=nn.LSTMCell):
        super(StackedRecurrentCells, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(rnn_cell(input_size, hidden_size, bias=bias))
            input_size = hidden_size

    def forward(self, inputs, hidden=None):
        def select_layer(h_state, i):  # To work on both LSTM / GRU, RNN
            if isinstance(h_state, tuple):
                return tuple([select_layer(s, i) for s in h_state])
            else:
                return h_state[i]

        if hidden is None:
            zeros = inputs.data.new(
                self.num_layers, inputs.size(0),  self.hidden_size).zero_()
            if isinstance(self.layers[0], nn.LSTMCell):
                hidden = (Variable(zeros, requires_grad=False),
                          Variable(zeros, requires_grad=False))
            else:
                hidden = Variable(zeros, requires_grad=False)
        next_hidden = []
        for i, layer in enumerate(self.layers):
            next_hidden_i = layer(inputs, select_layer(hidden, i))
            inputs = next_hidden_i[0] if isinstance(next_hidden_i, tuple) \
                else next_hidden_i
            if i + 1 != self.num_layers:
                inputs = self.dropout(inputs)
            next_hidden.append(next_hidden_i)
        if isinstance(hidden, tuple):
            next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
        else:
            next_hidden = torch.stack(next_hidden)
        return inputs, next_hidden


class StackedRecurrentAttention(nn.Module):

    def __init__(self, input_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False, context_size=None,
                 context_transform_size=None, dropout=0, rnn_cell=nn.LSTMCell, attention='bahdanau'):
        super(StackedRecurrentAttention, self).__init__()
        self.layers = num_layers
        if context_transform_size is not None:  # additional transform on context before attention
            self.context_transform = nn.Linear(
                context_size, context_transform_size)
            context_size = context_transform_size
        self.rnn = StackedRecurrentCells(input_size, hidden_size,
                                         num_layers=num_layers, bias=bias,
                                         batch_first=batch_first,
                                         dropout=dropout, rnn_cell=rnn_cell)
        self.attn = AttentionLayer(hidden_size, context_size, mode=attention,
                                   batch_first=batch_first, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, inputs, hidden, context, mask_attention=None, get_attention=False):
        outputs = []
        attentions = []
        self.attn.set_mask(mask_attention)
        if hasattr(self, 'context_transform'):
            context = self.context_transform(context)
        for input_t in inputs.split(1):
            input_t = input_t.squeeze(0)
            output_t, hidden = self.rnn(input_t, hidden)
            output_t, attn = self.attn(output_t, context)
            output_t = self.dropout(output_t)
            outputs += [output_t]
            if get_attention:
                attentions += [attn]

        outputs = torch.stack(outputs)
        if get_attention:
            attentions = torch.stack(attentions)
            return outputs, hidden, attentions
        else:
            return outputs, hidden


class RecurrentAttention(nn.Module):

    def __init__(self, input_size, context_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 attention_size=None, dropout=0, rnn=nn.LSTM, attention='bahdanau'):
        super(RecurrentAttention, self).__init__()
        self.layers = num_layers
        if attention_size is not None:  # additional transform on context before attention
            self.context_transform = nn.Linear(context_size, attention_size)
            context_size = attention_size
        self.rnn = rnn(input_size, hidden_size,
                       num_layers=num_layers, bias=bias,
                       batch_first=batch_first,
                       dropout=dropout)
        self.attn = AttentionLayer(hidden_size, context_size, mode=attention,
                                   batch_first=batch_first, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, inputs, hidden, context, mask_attention=None, get_attention=False):
        self.attn.set_mask(mask_attention)
        if hasattr(self, 'context_transform'):
            context = self.context_transform(context)
        outputs, hidden = self.rnn(inputs, hidden)
        outputs, attentions = self.attn(outputs, context)
        outputs = self.dropout(outputs)

        if get_attention:
            return outputs, hidden, attentions
        else:
            return outputs, hidden

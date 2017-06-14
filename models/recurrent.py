# Partially adapted from https://github.com/OpenNMT/OpenNMT-py

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from .seq2seq import Seq2Seq
from .attention import GlobalAttention

class RecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, rnn=nn.LSTM):
        super(RecurrentEncoder, self).__init__()
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        embedding_size = hidden_size
        if bidirectional:
            assert hidden_size % 2 == 0
            self.hidden_size = hidden_size // 2
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     padding_idx=0)
        self.rnn = rnn(embedding_size, self.hidden_size,
                       num_layers=num_layers, bias=bias,
                       batch_first=batch_first,
                       dropout=dropout, bidirectional=bidirectional)

    def forward(self, inputs, hidden=None):
        if isinstance(inputs, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = inputs[1].data.view(-1).tolist()
            emb = pack(self.embedder(inputs[0]), lengths)
        else:
            emb = self.embedder(inputs)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, tuple):
            outputs = unpack(outputs)[0]
        return outputs, hidden_t


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
            zeros = inputs.data.new(inputs.size(0), self.num_layers, self.hidden_size).zero_()
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


class RecurrentAttention(nn.Module):

    def __init__(self, input_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, rnn_cell=nn.LSTMCell, attention=GlobalAttention):
        super(RecurrentAttention, self).__init__()
        self.layers = num_layers
        self.rnn = StackedRecurrentCells(input_size, hidden_size,
                                         num_layers=num_layers, bias=bias,
                                         batch_first=batch_first,
                                         dropout=dropout, rnn_cell=rnn_cell)
        self.attn = attention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, inputs, hidden, context, get_attention=False):
        outputs = []
        attentions = []
        for input_t in inputs.split(1):
            input_t = input_t.squeeze(0)
            output_t, hidden = self.rnn(input_t, hidden)
            output_t, attn = self.attn(output_t, context.t())
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


class RecurrentAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, tie_embedding=False,
                 rnn_cell=nn.LSTMCell, attention=GlobalAttention):
        super(RecurrentAttentionDecoder, self).__init__()
        self.layers = num_layers

        self.embedder = nn.Embedding(vocab_size,
                                     hidden_size,
                                     padding_idx=0)
        self.rnn = RecurrentAttention(hidden_size, hidden_size,
                                      num_layers=num_layers, bias=bias, batch_first=batch_first,
                                      dropout=dropout, rnn_cell=rnn_cell, attention=attention)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

        self.hidden_size = hidden_size

    def forward(self, inputs, context, get_attention=False):
        context, hidden = context
        emb = self.embedder(inputs)
        if get_attention:
            x, hidden, attentions = self.rnn(
                emb, hidden, context, get_attention=get_attention)
        else:
            x, hidden = self.rnn(emb, hidden, context)
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)

        if get_attention:
            return x, (context, hidden), attentions
        else:
            return x, (context, hidden)


class RecurrentAttentionSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256,
                 num_layers=2, bias=True, dropout=0, tie_enc_dec_embedding=False):
        super(RecurrentAttentionSeq2Seq, self).__init__()
        self.encoder = RecurrentEncoder(vocab_size, hidden_size=hidden_size,
                                        num_layers=num_layers, bias=bias, dropout=dropout)
        self.decoder = RecurrentAttentionDecoder(vocab_size, hidden_size=hidden_size,
                                                 tie_embedding=tie_enc_dec_embedding,
                                                 num_layers=num_layers, bias=bias, dropout=dropout)

        if tie_enc_dec_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

    def bridge(self, context):
        context, hidden = context
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        new_hidden = []
        for h in hidden:
            if self.encoder.bidirectional:
                new_h = h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
            else:
                new_h = h
            new_hidden.append(new_h)
        return (context, tuple(new_hidden))

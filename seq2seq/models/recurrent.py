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


def bridge_bidirectional_hidden(hidden):
    #  the bidirectional hidden is  (layers*directions) x batch x dim
    #  we need to convert it to layers x batch x (directions*dim)
    num_layers = hidden.size(0) // 2
    batch_size, hidden_size = hidden.size(1), hidden.size(2)
    return hidden.view(num_layers, 2, batch_size, hidden_size) \
        .transpose(1, 2).contiguous() \
        .view(num_layers, batch_size, hidden_size * 2)


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
                                     padding_idx=PAD)
        self.rnn = rnn(embedding_size, self.hidden_size,
                       num_layers=num_layers, bias=bias,
                       batch_first=batch_first,
                       dropout=dropout, bidirectional=bidirectional)

    def forward(self, inputs, hidden=None):
        if isinstance(inputs, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = inputs[1].data.view(-1).tolist()
            padding_mask = inputs[0].data.eq(PAD)
            emb = pack(self.embedder(inputs[0]), lengths)
        else:
            padding_mask = inputs.data.eq(PAD)
            emb = self.embedder(inputs)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, tuple):
            outputs = unpack(outputs)[0]
        return outputs, hidden_t, padding_mask


class RecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, rnn=nn.LSTM, tie_embedding=True):
        super(RecurrentDecoder, self).__init__()
        self.layers = num_layers
        self.hidden_size = hidden_size
        embedding_size = hidden_size
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     padding_idx=PAD)
        self.rnn = rnn(embedding_size, self.hidden_size,
                       num_layers=num_layers, bias=bias,
                       batch_first=batch_first,
                       dropout=dropout, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

    def forward(self, inputs, hidden=None):
        if isinstance(inputs, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = inputs[1].data.view(-1).tolist()
            emb = pack(self.embedder(inputs[0]), lengths)
        else:
            emb = self.embedder(inputs)
        x, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, tuple):
            x = unpack(x)[0]
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)
        return x, hidden_t


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


class RecurrentAttention(nn.Module):

    def __init__(self, input_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False, context_size=None,
                 context_transform_size=None, dropout=0, rnn_cell=nn.LSTMCell, attention='bahdanau'):
        super(RecurrentAttention, self).__init__()
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
        if hasattr(self, 'context_transform'):
            context = self.context_transform(context)
        if mask_attention is not None:
            self.attn.set_mask(mask_attention)
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


class RecurrentAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, tie_embedding=False, context_size=None,
                 context_transform_size=None, rnn_cell=nn.LSTMCell, attention='bahdanau'):
        super(RecurrentAttentionDecoder, self).__init__()
        self.layers = num_layers

        self.embedder = nn.Embedding(vocab_size,
                                     hidden_size,
                                     padding_idx=PAD)
        self.rnn = RecurrentAttention(hidden_size, hidden_size,
                                      context_size=context_size, context_transform_size=context_transform_size,
                                      num_layers=num_layers, bias=bias, batch_first=batch_first,
                                      dropout=dropout, rnn_cell=rnn_cell, attention=attention)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

        self.hidden_size = hidden_size

    def forward(self, inputs, context, get_attention=False):
        context, hidden, padding_mask = context
        emb = self.embedder(inputs)
        if get_attention:
            x, hidden, attentions = self.rnn(
                emb, hidden, context, get_attention=get_attention)
        else:
            x, hidden = self.rnn(emb, hidden, context,
                                 mask_attention=padding_mask)
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)

        context = (context, hidden, padding_mask)
        if get_attention:
            return x, context, attentions
        else:
            return x, context


class RecurrentAttentionSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256, num_layers=2, embedding_size=None,
                 num_layers_decoder=None, num_layers_encoder=None,
                 bidirectional_encoder=True, bias=True, dropout=0,
                 attention='bahdanau', tie_embedding=False, transfer_hidden=True):
        super(RecurrentAttentionSeq2Seq, self).__init__()
        self.transfer_hidden = transfer_hidden
        embedding_size = embedding_size
        num_layers_encoder = num_layers_encoder or num_layers
        num_layers_decoder = num_layers_decoder or num_layers
        self.encoder = RecurrentEncoder(vocab_size, hidden_size=hidden_size, bidirectional=bidirectional_encoder,
                                        num_layers=num_layers_encoder, bias=bias, dropout=dropout)
        self.decoder = RecurrentAttentionDecoder(vocab_size, hidden_size=hidden_size,
                                                 tie_embedding=tie_embedding, attention=attention,
                                                 num_layers=num_layers_decoder, bias=bias, dropout=dropout)

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

    def bridge(self, context):
        context, hidden, padding_mask = context
        if not self.transfer_hidden:
            return (context, None, padding_mask)
        else:
            new_hidden = []
            for h in hidden:
                if self.encoder.bidirectional:
                    new_h = bridge_bidirectional_hidden(h)
                else:
                    new_h = h
                new_hidden.append(new_h)
            return (context, tuple(new_hidden), padding_mask)


class RecurrentLanguageModel(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256,
                 num_layers=2, bias=True, dropout=0, tie_embedding=False):
        super(RecurrentLanguageModel, self).__init__()
        self.decoder = RecurrentDecoder(vocab_size, hidden_size=hidden_size,
                                        tie_embedding=tie_embedding,
                                        num_layers=num_layers, bias=bias, dropout=dropout)

    def encode(self, *kargs, **kwargs):
        return None

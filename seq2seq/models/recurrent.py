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


def Recurrent(mode, input_size, hidden_size,
              num_layers=1, bias=True, batch_first=False,
              dropout=0, bidirectional=False, residual=False):
    params = dict(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers, bias=bias, batch_first=batch_first,
                  dropout=dropout, bidirectional=bidirectional)
    if mode == 'LSTM':
        rnn = nn.LSTM
    elif mode == 'GRU':
        rnn = nn.GRU
    else:
        raise Exception('Unknown mode: {}'.format(mode))
    if not residual:
        module = rnn(**params)
    else:
        module = StackedRecurrent(residual=True)
        params['num_layers'] = 1
        for i in range(num_layers):
            module.add_module(str(i), rnn(**params))

    return module


class StackedRecurrent(nn.Sequential):

    def __init__(self, residual=False):
        super(StackedRecurrent, self).__init__()
        self.residual = residual

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self))
        next_hidden = []
        for i, module in enumerate(self._modules.values()):
            output, h = module(inputs, hidden[i])
            next_hidden.append(h)
            if self.residual:
                inputs = output + inputs
            else:
                inputs = output
        return output, tuple(next_hidden)


class RecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, embedding_size=None,
                 num_layers=1, bias=True, batch_first=False, dropout=0,
                 bidirectional=False, num_bidirectional=None, mode='LSTM', residual=False):
        super(RecurrentEncoder, self).__init__()
        self.layers = num_layers
        self.bidirectional = bidirectional
        embedding_size = embedding_size or hidden_size
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     padding_idx=PAD)
        if bidirectional and num_bidirectional is None or num_bidirectional > 0:
            assert hidden_size % 2 == 0
            hidden_size = hidden_size // 2
        if num_bidirectional is not None and num_bidirectional < num_layers:
            self.rnn = StackedRecurrent()
            self.rnn.add_module('bidirectional', Recurrent(mode, embedding_size, hidden_size,
                                                           num_layers=num_bidirectional, bias=bias,
                                                           batch_first=batch_first, residual=residual,
                                                           dropout=dropout, bidirectional=True))
            self.rnn.add_module('unidirectional', Recurrent(mode, hidden_size * 2, hidden_size * 2,
                                                            num_layers=num_layers - num_bidirectional, bias=bias,
                                                            batch_first=batch_first, residual=residual,
                                                            dropout=dropout, bidirectional=False))
        else:
            self.rnn = Recurrent(mode, embedding_size, hidden_size,
                                 num_layers=num_layers, bias=bias,
                                 batch_first=batch_first, residual=residual,
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
                 dropout=0, mode='LSTM', residual=False, tie_embedding=True):
        super(RecurrentDecoder, self).__init__()
        self.layers = num_layers
        self.hidden_size = hidden_size
        embedding_size = hidden_size
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     padding_idx=PAD)
        self.rnn = Recurrent(mode, embedding_size, self.hidden_size,
                             num_layers=num_layers, bias=bias,
                             batch_first=batch_first, residual=residual,
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


class RecurrentAttention(nn.Module):

    def __init__(self, input_size, context_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False, dropout=0,
                 mode='LSTM', residual=False, context_transform=None, attention=None):
        super(RecurrentAttention, self).__init__()
        attention = attention or {}
        self.layers = num_layers
        if context_transform is not None:  # additional transform on context before attention
            self.context_transform = nn.Linear(context_size, context_transform)
            context_size = context_transform
        self.rnn = Recurrent(mode, input_size, hidden_size,
                             num_layers=num_layers, bias=bias,
                             batch_first=batch_first, residual=residual,
                             dropout=dropout)
        self.attn = AttentionLayer(hidden_size, context_size, batch_first=batch_first,
                                   **attention)
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


class RecurrentAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, hidden_size=128,
                 embedding_size=None, num_layers=1, bias=True,
                 batch_first=False, dropout=0, tie_embedding=False,
                 residual=False, context_transform=None, attention=None):
        super(RecurrentAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        attention = attention or {}

        self.layers = num_layers
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     padding_idx=PAD)
        self.rnn = RecurrentAttention(hidden_size, context_size, hidden_size, num_layers=num_layers,
                                      bias=bias, batch_first=batch_first, dropout=dropout,
                                      context_transform=context_transform, residual=residual,
                                      attention=attention)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

        self.hidden_size = hidden_size

    def forward(self, inputs, context, get_attention=False):
        context, hidden, padding_mask = context
        emb = self.embedder(inputs)
        if get_attention:
            x, hidden, attentions = self.rnn(emb, hidden, context,
                                             mask_attention=padding_mask,
                                             get_attention=get_attention)
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
                 bias=True, dropout=0, tie_embedding=False, transfer_hidden=True,
                 residual=False, encoder=None, decoder=None):
        super(RecurrentAttentionSeq2Seq, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('num_layers', num_layers)
        encoder.setdefault('bias', bias)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('dropout', dropout)
        encoder.setdefault('residual', residual)

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('bias', bias)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('residual', residual)
        decoder['context_size'] = encoder['hidden_size']

        self.encoder = RecurrentEncoder(**encoder)
        self.decoder = RecurrentAttentionDecoder(**decoder)
        self.transfer_hidden = transfer_hidden

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

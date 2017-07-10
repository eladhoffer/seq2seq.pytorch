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


class StackedRecurrent(nn.Sequential):

    def __init__(self, *modules):
        super(StackedRecurrent, self).__init__(*modules)

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self))
        next_hidden = []
        for i, module in enumerate(self._modules.values()):
            output, h = module(inputs, hidden[i])
            next_hidden.append(h)
        return output, tuple(next_hidden)


class RecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, embedding_size=None,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, num_bidirectional=None, rnn=nn.LSTM):
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
            self.rnn.add_module('bidirectional', rnn(embedding_size, hidden_size,
                                                     num_layers=num_bidirectional, bias=bias,
                                                     batch_first=batch_first,
                                                     dropout=dropout, bidirectional=True))
            self.rnn.add_module('unidirectional', rnn(hidden_size * 2, hidden_size * 2,
                                                      num_layers=num_layers - num_bidirectional, bias=bias,
                                                      batch_first=batch_first,
                                                      dropout=dropout, bidirectional=False))
        else:
            self.rnn = rnn(embedding_size, hidden_size,
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


class RecurrentAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, hidden_size=128,
                 embedding_size=None, num_layers=1, bias=True,
                 batch_first=False, dropout=0, tie_embedding=False,
                 attention_size=None, attention='bahdanau'):
        super(RecurrentAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.layers = num_layers

        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     padding_idx=PAD)
        self.rnn = RecurrentAttention(hidden_size, context_size, hidden_size,
                                      attention_size=attention_size,
                                      num_layers=num_layers, bias=bias, batch_first=batch_first,
                                      dropout=dropout, attention=attention)
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
                 num_layers_decoder=None, num_layers_encoder=None,
                 embedding_size_encoder=None, embedding_size_decoder=None,
                 hidden_size_encoder=None, hidden_size_decoder=None, bias=True, dropout=0, tie_embedding=False,
                 bidirectional_encoder=True,  num_bidirectional=None,
                 attention='bahdanau', attention_size=None, transfer_hidden=True):
        super(RecurrentAttentionSeq2Seq, self).__init__()
        self.transfer_hidden = transfer_hidden
        embedding_size = embedding_size or hidden_size
        num_layers_encoder = num_layers_encoder or num_layers
        num_layers_decoder = num_layers_decoder or num_layers
        embedding_size_encoder = embedding_size_encoder or embedding_size
        embedding_size_decoder = embedding_size_decoder or embedding_size
        hidden_size_encoder = hidden_size_encoder or hidden_size
        hidden_size_decoder = hidden_size_decoder or hidden_size

        self.encoder = RecurrentEncoder(vocab_size, hidden_size=hidden_size_encoder,
                                        embedding_size=embedding_size_encoder, bidirectional=bidirectional_encoder,
                                        num_bidirectional=num_bidirectional, num_layers=num_layers_encoder, bias=bias, dropout=dropout)
        self.decoder = RecurrentAttentionDecoder(vocab_size, hidden_size_encoder, hidden_size=hidden_size,
                                                 embedding_size=embedding_size_decoder, tie_embedding=tie_embedding,
                                                 attention=attention, attention_size=attention_size,
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

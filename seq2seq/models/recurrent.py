import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from .seq2seq_base import Seq2Seq
from .modules.recurrent import Recurrent, RecurrentAttention, StackedRecurrent
from .modules.state import State
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

    def __init__(self, vocab_size, hidden_size=128, embedding_size=None,
                 num_layers=1, bias=True, batch_first=False, dropout=0,
                 bidirectional=False, num_bidirectional=None, mode='LSTM', residual=False):
        super(RecurrentEncoder, self).__init__()
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        embedding_size = embedding_size or hidden_size
        num_bidirectional = num_bidirectional or num_layers
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     padding_idx=PAD)
        if bidirectional and num_bidirectional > 0:
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

        state = State(outputs=outputs, hidden=hidden_t,
                      mask=padding_mask, batch_first=self.batch_first)
        return state


class RecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, mode='LSTM', residual=False, tie_embedding=True):
        super(RecurrentDecoder, self).__init__()
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first
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

    def forward(self, inputs, state):
        hidden = state.hidden
        if isinstance(inputs, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = inputs[1].data.view(-1).tolist()
            emb = pack(self.embedder(inputs[0]), lengths)
        else:
            emb = self.embedder(inputs)
        x, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, tuple):
            x = unpack(x)[0]

        x = self.classifier(x)
        return x, State(hidden=hidden_t, batch_first=self.batch_first)


class RecurrentAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, hidden_size=128,
                 embedding_size=None, num_layers=1, bias=True,
                 batch_first=False, dropout=0, tie_embedding=False,
                 residual=False, context_transform=None, attention=None,
                 concat_attention=True, num_pre_attention_layers=None):
        super(RecurrentAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        attention = attention or {}

        self.layers = num_layers
        self.batch_first = batch_first
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     padding_idx=PAD)
        self.rnn = RecurrentAttention(hidden_size, context_size, hidden_size, num_layers=num_layers,
                                      bias=bias, batch_first=batch_first, dropout=dropout,
                                      context_transform=context_transform, residual=residual,
                                      attention=attention, concat_attention=concat_attention,
                                      num_pre_attention_layers=num_pre_attention_layers)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.classifier.weight = self.embedder.weight

        self.hidden_size = hidden_size

    def forward(self, inputs, state, get_attention=False):
        context, hidden = state.context, state.hidden
        emb = self.embedder(inputs)
        if get_attention:
            x, hidden, attentions = self.rnn(emb, context.outputs, state.hidden,
                                             mask_attention=context.mask,
                                             get_attention=get_attention)
        else:
            x, hidden = self.rnn(emb, context.outputs, state.hidden,
                                 mask_attention=context.mask)
        x = self.classifier(x)

        new_state = State(hidden=hidden, context=context, batch_first=self.batch_first)
        if get_attention:
            new_state.attention_score = attentions
        return x, new_state


class RecurrentAttentionSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256, num_layers=2, embedding_size=None,
                 bias=True, dropout=0, tie_embedding=False, transfer_hidden=True,
                 residual=False, encoder=None, decoder=None, batch_first=False):
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
        encoder.setdefault('batch_first', batch_first)

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('bias', bias)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('residual', residual)
        decoder.setdefault('batch_first', batch_first)
        decoder['context_size'] = encoder['hidden_size']

        self.encoder = RecurrentEncoder(**encoder)
        self.decoder = RecurrentAttentionDecoder(**decoder)
        self.transfer_hidden = transfer_hidden

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

    def bridge(self, context):
        state = State(context=context, batch_first=self.decoder.batch_first)
        if not self.transfer_hidden:
            state.hidden = None
        else:
            hidden = state.context.hidden
            new_hidden = []
            for h in hidden:
                if self.encoder.bidirectional:
                    new_h = bridge_bidirectional_hidden(h)
                else:
                    new_h = h
                new_hidden.append(new_h)
            state.hidden = new_hidden
        return state


class RecurrentLanguageModel(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256,
                 num_layers=2, bias=True, dropout=0, tie_embedding=False):
        super(RecurrentLanguageModel, self).__init__()
        self.decoder = RecurrentDecoder(vocab_size, hidden_size=hidden_size,
                                        tie_embedding=tie_embedding,
                                        num_layers=num_layers, bias=bias, dropout=dropout)

    def encode(self, *kargs, **kwargs):
        return None

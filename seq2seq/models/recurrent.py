import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from .seq2seq_base import Seq2Seq
from .modules.recurrent import Recurrent, RecurrentAttention, StackedRecurrent
from .modules.state import State
from .modules.weight_norm import weight_norm as wn
from seq2seq.tools.config import PAD


def bridge_bidirectional_hidden(hidden, hidden_output_size):
    #  the bidirectional hidden is  (layers*directions) x batch x dim
    #  we need to convert it to layers x batch x (directions*dim)
    if hidden_output_size == hidden.size(-1):
        return hidden
    else:  # bidirectional with halved size
        num_layers = hidden.size(0) // 2
        batch_size, hidden_size = hidden.size(1), hidden.size(2)
        return hidden.view(num_layers, 2, batch_size, hidden_size) \
            .transpose(1, 2).contiguous() \
            .view(num_layers, batch_size, hidden_size * 2)


class HiddenTransform(nn.Module):
    """docstring for [object Object]."""

    def __init__(self, input_shape, output_shape, activation='tanh', bias=True, batch_first=False):
        super(HiddenTransform, self).__init__()
        self.batch_first = batch_first
        self.activation = nn.Tanh() if activation == 'tanh' else None
        if not isinstance(input_shape, tuple):
            input_shape = (input_shape,)
        if not isinstance(output_shape, tuple):
            output_shape = (output_shape,)
        assert len(input_shape) == len(output_shape)
        self.module_list = nn.ModuleList()
        for i in range(len(input_shape)):
            self.module_list.append(
                nn.Linear(input_shape[i], output_shape[i], bias=bias))

    def forward(self, hidden):
        hidden_in = hidden if isinstance(hidden, tuple)\
            else (hidden,)
        hidden_out = []
        for i, h_in in enumerate(hidden_in):
            if not self.batch_first:
                h_in = h_in.transpose(0, 1)
            h_in = h_in.contiguous().view(h_in.size(0), -1)
            h_out = self.module_list[i](h_in)
            if self.activation is not None:
                h_out = self.activation(h_out)
            h_out = h_out.unsqueeze(1 if self.batch_first else 0)
            hidden_out.append(h_out)
        if isinstance(hidden, tuple):
            return tuple(hidden_out)
        else:
            return hidden_out[0]


class RecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, embedding_size=None, num_layers=1,
                 bias=True, batch_first=False, dropout=0, embedding_dropout=0, forget_bias=None,
                 context_transform=None, context_transform_bias=True, hidden_transform=None, hidden_transform_bias=True,
                 bidirectional=True, adapt_bidirectional_size=False, num_bidirectional=None,
                 mode='LSTM', pack_inputs=True, residual=False, weight_norm=False):
        super(RecurrentEncoder, self).__init__()
        self.pack_inputs = pack_inputs  # pack input using PackedSequence
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        embedding_size = embedding_size or hidden_size
        num_bidirectional = num_bidirectional or num_layers
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     sparse=False,
                                     padding_idx=PAD)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)

        if adapt_bidirectional_size and bidirectional and num_bidirectional > 0:
            # adapt hidden size to have output size same as input
            assert hidden_size % 2 == 0
            hidden_size = hidden_size // 2
        self.hidden_size = hidden_size
        self.context_size = 2 * hidden_size if bidirectional else hidden_size
        if context_transform is not None:  # additional transform on context before output
            self.context_transform = nn.Linear(self.context_size,
                                               context_transform, bias=context_transform_bias)
            self.context_size = (context_transform, self.context_size)
            if weight_norm:
                self.context_transform = wn(self.context_transform)
        if num_bidirectional is not None and num_bidirectional < num_layers:
            assert hidden_transform is None, "hidden transform can be used only for single bidi encoder for now"
            self.rnn = StackedRecurrent(dropout=dropout, residual=residual)
            self.rnn.add_module('bidirectional', Recurrent(mode, embedding_size, hidden_size,
                                                           num_layers=num_bidirectional, bias=bias,
                                                           batch_first=batch_first, residual=residual,
                                                           weight_norm=weight_norm, forget_bias=forget_bias,
                                                           dropout=dropout, bidirectional=True))
            self.rnn.add_module('unidirectional', Recurrent(mode, hidden_size * 2, hidden_size * 2,
                                                            num_layers=num_layers - num_bidirectional,
                                                            batch_first=batch_first, residual=residual,
                                                            weight_norm=weight_norm, forget_bias=forget_bias,
                                                            bias=bias, dropout=dropout, bidirectional=False))
        else:
            self.rnn = Recurrent(mode, embedding_size, hidden_size,
                                 num_layers=num_layers, bias=bias,
                                 batch_first=batch_first, residual=residual,
                                 weight_norm=weight_norm, dropout=dropout,
                                 bidirectional=bidirectional)
            if hidden_transform is not None:
                hidden_size = hidden_size * num_layers
                if bidirectional:
                    hidden_size += hidden_size
                if mode == 'LSTM':
                    hidden_size = (hidden_size, hidden_size)

                self.hidden_transform = HiddenTransform(
                    hidden_size, hidden_transform, bias=hidden_transform_bias, batch_first=batch_first)

    def forward(self, inputs, hidden=None):
        if isinstance(inputs, PackedSequence):
            emb = PackedSequence(self.embedding_dropout(
                self.embedder(inputs.data)), inputs.batch_sizes)
            bsizes = inputs.batch_sizes.to(device=inputs.data.device)
            max_batch = int(bsizes[0])
            # Get padding mask
            time_dim = 1 if self.batch_first else 0
            range_batch = torch.arange(0, max_batch,
                                       dtype=bsizes.dtype,
                                       device=bsizes.device)
            range_batch = range_batch.unsqueeze(time_dim)
            bsizes = bsizes.unsqueeze(1 - time_dim)
            padding_mask = (bsizes - range_batch).le(0)
        else:
            padding_mask = inputs.eq(PAD)
            emb = self.embedding_dropout(self.embedder(inputs))
        outputs, hidden_t = self.rnn(emb, hidden)

        if isinstance(inputs, PackedSequence):
            outputs = unpack(outputs)[0]
        outputs = self.dropout(outputs)
        if hasattr(self, 'context_transform'):
            context = self.context_transform(outputs)
        else:
            context = None

        if hasattr(self, 'hidden_transform'):
            hidden_t = self.hidden_transform(hidden_t)

        state = State(outputs=outputs, hidden=hidden_t, context=context,
                      mask=padding_mask, batch_first=self.batch_first)
        return state


class RecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, hidden_size=128, embedding_size=None, num_layers=1,
                 bias=True, forget_bias=None, batch_first=False,  dropout=0, embedding_dropout=0,
                 mode='LSTM', residual=False, weight_norm=False, tie_embedding=True):
        super(RecurrentDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     sparse=False,
                                     padding_idx=PAD)
        self.rnn = Recurrent(mode, embedding_size, self.hidden_size,
                             num_layers=num_layers, bias=bias, forget_bias=forget_bias,
                             batch_first=batch_first, residual=residual, weight_norm=weight_norm,
                             dropout=dropout, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)

        if tie_embedding:
            self.classifier.weight = self.embedder.weight

    def forward(self, inputs, state):
        context, hidden = state.context, state.hidden
        if isinstance(inputs, PackedSequence):
            emb = PackedSequence(self.embedding_dropout(
                self.embedder(inputs.data)), inputs.batch_size)
        else:
            emb = self.embedding_dropout(self.embedder(inputs))
        x, hidden_t = self.rnn(emb, hidden)
        if isinstance(inputs, PackedSequence):
            x = unpack(x)[0]
        x = self.dropout(x)

        x = self.classifier(x)
        return x, State(hidden=hidden_t, context=context, batch_first=self.batch_first)


class RecurrentAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, hidden_size=128, embedding_size=None,
                 num_layers=1, bias=True, forget_bias=None, batch_first=False, bias_classifier=True,
                 dropout=0, embedding_dropout=0, tie_embedding=False, residual=False, mode='LSTM',
                 weight_norm=False, attention=None, concat_attention=True, num_pre_attention_layers=None):
        super(RecurrentAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        attention = attention or {}

        self.layers = num_layers
        self.batch_first = batch_first
        self.embedder = nn.Embedding(vocab_size,
                                     embedding_size,
                                     sparse=False,
                                     padding_idx=PAD)
        self.rnn = RecurrentAttention(embedding_size, context_size, hidden_size, num_layers=num_layers,
                                      bias=bias, batch_first=batch_first, dropout=dropout, mode=mode,
                                      forget_bias=forget_bias, residual=residual, weight_norm=weight_norm,
                                      attention=attention, concat_attention=concat_attention,
                                      num_pre_attention_layers=num_pre_attention_layers)
        self.classifier = nn.Linear(
            hidden_size, vocab_size, bias=bias_classifier)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)

        if tie_embedding:
            self.classifier.weight = self.embedder.weight

        self.hidden_size = hidden_size

    def forward(self, inputs, state, get_attention=False):
        context, hidden = state.context, state.hidden
        if context.context is not None:
            attn_input = (context.context, context.outputs)
        else:
            attn_input = context.outputs
        emb = self.embedding_dropout(self.embedder(inputs))

        if get_attention:
            x, hidden, attentions = self.rnn(emb, attn_input, state.hidden,
                                             mask_attention=context.mask,
                                             get_attention=get_attention)
        else:
            x, hidden = self.rnn(emb, attn_input, state.hidden,
                                 mask_attention=context.mask)
        x = self.dropout(x)
        x = self.classifier(x)

        new_state = State(hidden=hidden, context=context,
                          batch_first=self.batch_first)
        if get_attention:
            new_state.attention_score = attentions
        return x, new_state


class RecurrentAttentionSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256, num_layers=2,
                 embedding_size=None, bias=True, dropout=0, embedding_dropout=0,
                 tie_embedding=False, transfer_hidden=False, forget_bias=None, bias_classifier=True,
                 residual=False, weight_norm=False, encoder=None, decoder=None, batch_first=False):
        super(RecurrentAttentionSeq2Seq, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('bidirectional', True)
        encoder.setdefault('num_layers', num_layers)
        encoder.setdefault('bias', bias)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('dropout', dropout)
        encoder.setdefault('embedding_dropout', embedding_dropout)
        encoder.setdefault('residual', residual)
        encoder.setdefault('weight_norm', weight_norm)
        encoder.setdefault('batch_first', batch_first)
        encoder.setdefault('forget_bias', forget_bias)
        if not transfer_hidden:  # no use for hidden transform if not transferred
            encoder['hidden_transform'] = None

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('bias', bias)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('embedding_dropout', embedding_dropout)
        decoder.setdefault('residual', residual)
        decoder.setdefault('weight_norm', weight_norm)
        decoder.setdefault('batch_first', batch_first)
        decoder.setdefault('forget_bias', forget_bias)
        decoder.setdefault('bias_classifier', bias_classifier)

        self.encoder = RecurrentEncoder(**encoder)
        decoder['context_size'] = self.encoder.context_size

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
                    new_h = bridge_bidirectional_hidden(h,
                                                        self.decoder.hidden_size)
                else:
                    new_h = h
                new_hidden.append(new_h)
            state.hidden = tuple(new_hidden)
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

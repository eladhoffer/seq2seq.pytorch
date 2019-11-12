import torch
import torch.nn as nn
import math
from random import randrange, shuffle
from copy import deepcopy
from .seq2seq_base import Seq2Seq
from seq2seq.tools.config import PAD, EOS
from .modules.state import State
from .modules.transformer_blocks import EncoderBlock, DecoderBlock, EncoderBlockPreNorm, DecoderBlockPreNorm, positional_embedding, CharWordEmbedder, PositionalEmbedding, MultiHeadAttention
from .modules.attention import SDPAttention


def index_select_2d(x, order):
    idxs = order.add(torch.arange(order.size(0), dtype=torch.long,
                                  device=order.device).view(-1, 1) * order.size(1))
    out_sz = x.shape
    out_sz = (order.size(0), order.size(1), *out_sz[2:])
    return x.flatten(0, 1).index_select(0, idxs.contiguous().view(-1)).view(out_sz)


@torch.jit.script
def _reorder(order):
    # type: (Tensor) -> Tensor
    B, T = order.shape
    reorder_list = []

    for j in range(T):
        reorder_list.append(order.eq(j).nonzero()[:, -1])
    return torch.stack(reorder_list, dim=-1)

# @torch.jit.script


def rand_order(T, block_size=None, block_ratio=0.25, out=None):
    if block_size is None:
        block_size = max(int(round(T * block_ratio)), 1)
    if block_size == 1:
        return torch.randperm(T, out=out)
    else:
        if out is None:
            out = torch.empty((T,), dtype=torch.long)
        # torch.arange(T, out=out)
        order = list(range(T))
        offset = randrange(T)
        order = torch.tensor(order[offset:] + order[:offset])
        order = list(order.split(block_size))
        shuffle(order)
        order = torch.cat(order)
        out.copy_(order)
    return out


def permuted_order(inputs, padding_idx=PAD, eos_idx=EOS, batch_first=True):
    # type: (Tensor, int, int, bool) -> Tuple[Tensor, Tensor]
    time_dim, batch_dim = (1, 0) if batch_first else (0, 1)
    B, T = inputs.size(batch_dim), inputs.size(time_dim)
    order = torch.arange(-1, T, dtype=torch.long, device=inputs.device)
    order = order.view(1, -1).expand(B, T+1).contiguous()
    max_time = inputs.ne(padding_idx).sum(time_dim) - 1
    for i in range(B):
        t = int(max_time[i])
        scope = order[i].narrow(0, 1, t)
        rand_order(t, out=scope)
    order.add_(1)

    reorder = _reorder(order.narrow(time_dim, 1, T) - 1)
    if not batch_first:
        order = order.t()
        reorder = reorder.t()
    return order, reorder


def repeat(x, N, dim=0):
    # type: (Tensor, int, int) -> Tensor
    if x is None:
        return None
    sz = list(x.shape)
    expand_sz = list(x.shape)
    sz.insert(dim, 1)
    expand_sz.insert(dim, N)
    x = x.view(*sz)
    x = x.expand(*expand_sz)
    x = x.contiguous()  # .flatten(0, 1)
    return x


class TransformerAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None,
                 num_layers=6, num_heads=8, inner_linear=2048, inner_groups=1, prenormalized=False,
                 mask_symbol=PAD, batch_first=True, layer_norm=True, weight_norm=False, dropout=0, embedder=None):

        super(TransformerAttentionEncoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        if embedding_size != hidden_size:
            self.input_projection = nn.Parameter(
                torch.empty(embedding_size, hidden_size))
            nn.init.kaiming_uniform_(self.input_projection, a=math.sqrt(5))
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.mask_symbol = mask_symbol
        self.embedder = embedder or nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        if prenormalized:
            block = EncoderBlockPreNorm
        else:
            block = EncoderBlock
        self.blocks = nn.ModuleList([block(hidden_size,
                                           num_heads=num_heads,
                                           inner_linear=inner_linear,
                                           inner_groups=inner_groups,
                                           layer_norm=layer_norm,
                                           weight_norm=weight_norm,
                                           batch_first=batch_first,
                                           dropout=dropout)
                                     for _ in range(num_layers)
                                     ])
        if layer_norm and prenormalized:
            self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, hidden=None):
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        if hasattr(self, 'input_projection'):
            x = x @ self.input_projection
        pos_embedding = positional_embedding(x.size(time_dim), x.size(-1),
                                             device=x.device)
        x.add_(pos_embedding.unsqueeze(batch_dim))
        x = self.dropout(x)

        for block in self.blocks:
            block.set_mask(padding_mask)
            x = block(x)

        if hasattr(self, 'lnorm'):
            x = self.lnorm(x)

        return State(outputs=x, mask=padding_mask, batch_first=self.batch_first)


class TransformerAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6, num_heads=8,
                 batch_first=True, dropout=0, inner_linear=2048, inner_groups=1, prenormalized=False, stateful=None, state_dim=None,
                 mask_symbol=PAD, tie_embedding=True, layer_norm=True, weight_norm=False, embedder=None, classifier=True, permuted=False, learned_condition=False, max_length=512, **kwargs):

        super(TransformerAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        if embedding_size != hidden_size:
            self.input_projection = nn.Parameter(
                torch.empty(embedding_size, hidden_size))
            nn.init.kaiming_uniform_(self.input_projection, a=math.sqrt(5))
        self.batch_first = batch_first
        self.mask_symbol = mask_symbol
        self.embedder = embedder or nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.stateful = stateful
        self.permuted = permuted
        block_args = dict(hidden_size=hidden_size,
                          num_heads=num_heads,
                          inner_linear=inner_linear,
                          inner_groups=inner_groups,
                          layer_norm=layer_norm,
                          weight_norm=weight_norm,
                          dropout=dropout,
                          batch_first=batch_first,
                          stateful=stateful,
                          state_dim=state_dim)
        if permuted:
            if learned_condition:
                self.conditioned_pos = nn.Embedding(max_length, embedding_size)
            else:
                self.conditioned_pos = PositionalEmbedding(embedding_size,
                                                           min_timescale=1.0e4,
                                                           max_timescale=1.0e8)

        if prenormalized:
            block = DecoderBlockPreNorm
        else:
            block = DecoderBlock

        self.blocks = nn.ModuleList([block(**block_args)
                                     for _ in range(num_layers)])
        if layer_norm and prenormalized:
            self.lnorm = nn.LayerNorm(hidden_size)

        if classifier:
            self.classifier = nn.Linear(embedding_size, vocab_size)
            if tie_embedding:
                self.embedder.weight = self.classifier.weight

            if embedding_size != hidden_size:
                if tie_embedding:
                    self.output_projection = self.input_projection
                else:
                    self.output_projection = nn.Parameter(
                        torch.empty(embedding_size, hidden_size))
                    nn.init.kaiming_uniform_(
                        self.output_projection, a=math.sqrt(5))

    def forward(self, inputs, state, time_multiply=1, get_attention=False, causal=None,
                input_order=None, output_order=None, output_reorder=None):
        context = state.context
        time_step = 0
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)

        if self.stateful:
            block_state = state.hidden
            if block_state is None:
                self.time_step = 0
            time_step = self.time_step
        else:
            block_state = state.inputs
            time_step = 0 if block_state is None else \
                block_state[0][0].size(time_dim)
        if block_state is None:
            block_state = [None] * len(self.blocks)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None

        x = self.embedder(inputs).mul_(self.scale_embedding)

        if hasattr(self, 'input_projection'):
            x = x @ self.input_projection

        pos_embedding = positional_embedding(x.size(time_dim), x.size(-1), offset=time_step,
                                             device=x.device).unsqueeze(batch_dim)
        x.add_(pos_embedding)

        if self.permuted:
            if self.training:
                output_order, output_reorder = permuted_order(inputs,
                                                              batch_first=self.batch_first)
                pos_target = output_order.narrow(time_dim, 1, x.size(time_dim))

                pos_input = output_order.narrow(time_dim, 0, x.size(time_dim))
                x = index_select_2d(x, pos_input)

                cond_embedding = self.conditioned_pos(pos_target)

            else:
                pos_target = torch.arange(x.size(time_dim) * time_multiply,
                                          device=x.device) + time_step + 1
                cond_embedding = self.conditioned_pos(pos_target)\
                    .unsqueeze(batch_dim)
                output_reorder = None

            if time_multiply > 1:
                padding_mask = repeat(
                    padding_mask, time_multiply).flatten(0, 1)
                x = repeat(x, time_multiply).flatten(0, 1)
                cond_embedding = repeat(cond_embedding.squeeze(batch_dim), inputs.size(batch_dim), dim=1).transpose(0, 1)\
                    .contiguous().view_as(x)
                context.mask = repeat(
                    context.mask, time_multiply).flatten(0, 1)
                context.outputs = repeat(
                    context.outputs, time_multiply).flatten(0, 1)
            x = x.add(cond_embedding)

        x = self.dropout(x)

        attention_scores = []
        updated_state = []
        for i, block in enumerate(self.blocks):
            if causal is not None:
                block.masked_attention.causal = causal
            block.set_mask(padding_mask, context.mask)
            x, attn_enc, block_s = block(
                x, context.outputs, block_state[i])
            updated_state.append(block_s)
            if get_attention:
                attention_scores.append(attn_enc)
            else:
                del attn_enc

        if hasattr(self, 'lnorm'):
            x = self.lnorm(x)

        if output_reorder is not None:
            x = index_select_2d(x, output_reorder)

        if hasattr(self, 'output_projection'):
            x = x @ self.output_projection.t()

        if self.classifier is not None:
            x = self.classifier(x)

        if self.stateful:
            state.hidden = tuple(updated_state)
            self.time_step += 1
        else:
            state.inputs = tuple(updated_state)
        if get_attention:
            state.attention_score = attention_scores

        if time_multiply > 1:
            x = x.view(time_multiply, inputs.size(batch_dim), -1)
            x = x.transpose(0, 1)

        return x, state


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6, num_heads=8,
                 inner_linear=2048, inner_groups=1, dropout=0.1, prenormalized=False, tie_embedding=True,
                 encoder=None, decoder=None, layer_norm=True, weight_norm=False, batch_first=True, stateful=None):
        super(Transformer, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder = deepcopy(encoder)
        decoder = deepcopy(decoder)
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('num_layers', num_layers)
        encoder.setdefault('num_heads', num_heads)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('layer_norm', layer_norm)
        encoder.setdefault('weight_norm', weight_norm)
        encoder.setdefault('dropout', dropout)
        encoder.setdefault('inner_linear', inner_linear)
        encoder.setdefault('inner_groups', inner_groups)
        encoder.setdefault('prenormalized', prenormalized)
        encoder.setdefault('batch_first', batch_first)

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('num_heads', num_heads)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('layer_norm', layer_norm)
        decoder.setdefault('weight_norm', weight_norm)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('inner_linear', inner_linear)
        decoder.setdefault('inner_groups', inner_groups)
        decoder.setdefault('batch_first', batch_first)
        decoder.setdefault('prenormalized', prenormalized)
        decoder.setdefault('stateful', stateful)

        if isinstance(vocab_size, tuple):
            embedder = CharWordEmbedder(
                vocab_size[1], embedding_size, hidden_size)
            encoder.setdefault('embedder', embedder)
            decoder.setdefault('embedder', embedder)
            decoder['classifier'] = False

        self.batch_first = batch_first
        self.encoder = TransformerAttentionEncoder(**encoder)
        self.decoder = TransformerAttentionDecoder(**decoder)

        if tie_embedding and not isinstance(vocab_size, tuple):
            assert self.encoder.embedder.weight.shape == self.decoder.classifier.weight.shape
            self.encoder.embedder.weight = self.decoder.classifier.weight
            if embedding_size != hidden_size:
                self.encoder.input_projection = self.decoder.input_projection

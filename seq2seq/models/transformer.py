import torch
import torch.nn as nn
from .seq2seq_base import Seq2Seq
from seq2seq.tools.config import PAD
from .modules.state import State
from .modules.transformer_blocks import EncoderBlock, DecoderBlock, positional_embedding


class TransformerAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None,
                 num_layers=6, num_heads=8, inner_linear=2048, inner_groups=1,
                 mask_symbol=PAD, layer_norm=True, weight_norm=False, dropout=0):

        super(TransformerAttentionEncoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.hidden_size = hidden_size
        self.batch_first = True
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.blocks = nn.ModuleList([EncoderBlock(hidden_size,
                                                  num_heads=num_heads,
                                                  inner_linear=inner_linear,
                                                  layer_norm=layer_norm,
                                                  weight_norm=weight_norm,
                                                  dropout=dropout)
                                     for _ in range(num_layers)
                                     ])

    def forward(self, inputs, hidden=None):
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        x.add_(positional_embedding(x))
        x = self.dropout(x)

        for block in self.blocks:
            block.set_mask(padding_mask)
            x = block(x)

        return State(outputs=x, mask=padding_mask, batch_first=True)


class TransformerAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None,
                 num_layers=6, num_heads=8, dropout=0, inner_linear=2048, inner_groups=1, stateful=False,
                 mask_symbol=PAD, tie_embedding=True, layer_norm=True, weight_norm=False):

        super(TransformerAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.batch_first = True
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.stateful = stateful
        self.blocks = nn.ModuleList([DecoderBlock(hidden_size,
                                                  num_heads=num_heads,
                                                  inner_linear=inner_linear,
                                                  inner_groups=inner_groups,
                                                  layer_norm=layer_norm,
                                                  weight_norm=weight_norm,
                                                  dropout=dropout,
                                                  stateful=stateful)
                                     for _ in range(num_layers)
                                     ])
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.embedder.weight = self.classifier.weight

    def forward(self, inputs, state, get_attention=False):
        context = state.context
        time_step = 0
        if self.stateful:
            block_state = state.hidden
            if block_state is None:
                self.time_step = 0
            time_step = self.time_step
        else:
            block_state = state.inputs
            time_step = 0 if block_state is None else block_state[0].size(1)

        if block_state is None:
            block_state = [None] * len(self.blocks)

        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        x.add_(positional_embedding(x, offset=time_step))
        x = self.dropout(x)

        attention_scores = []
        updated_state = []
        for i, block in enumerate(self.blocks):
            block.set_mask(padding_mask, context.mask)
            x, attn_enc, block_s = block(x, context.outputs, block_state[i])
            updated_state.append(block_s)
            if get_attention:
                attention_scores.append(attn_enc)
            else:
                del attn_enc
        x = self.classifier(x)
        if self.stateful:
            state.hidden = tuple(updated_state)
            self.time_step += 1
        else:
            state.inputs = tuple(updated_state)
        if get_attention:
            state.attention_score = attention_scores
        return x, state


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6,
                 num_heads=8, inner_linear=2048, inner_groups=1, dropout=0.1, tie_embedding=True,
                 encoder=None, decoder=None, layer_norm=True, weight_norm=False, stateful=None):
        super(Transformer, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
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
        decoder.setdefault('stateful', stateful)

        self.batch_first = True
        self.encoder = TransformerAttentionEncoder(**encoder)
        self.decoder = TransformerAttentionDecoder(**decoder)

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.classifier.weight

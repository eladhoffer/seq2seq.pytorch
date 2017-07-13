import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .modules import LayerNorm1d
from .attention import MultiHeadAttention
from .seq2seq_base import Seq2Seq
from seq2seq.tools.config import PAD


def positional_embedding(x, min_timescale=1.0, max_timescale=1.0e4):
    batch, length, channels = list(x.size())
    assert (channels % 2 == 0)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (float(num_timescales) - 1.))
    position = torch.arange(0, length).float()
    inv_timescales = torch.arange(0, num_timescales).float()
    if x.is_cuda:
        position = position.cuda()
        inv_timescales = inv_timescales.cuda()

    inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
    scaled_time = position.unsqueeze(1).expand(
        length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
    # scaled time is now length x num_timescales
    # length x channels
    signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)
    return signal.unsqueeze(0).expand(batch, length, channels)


class EncoderBlock(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, inner_linear=1024, dropout=0):

        super(EncoderBlock, self).__init__()
        self.lnorm1 = LayerNorm1d(hidden_size)
        self.lnorm2 = LayerNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout=dropout, causal=False)
        self.fc = nn.Sequential(nn.Linear(hidden_size, inner_linear),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(inner_linear, hidden_size))

    def set_mask(self, mask):
        self.attention.set_mask_q(mask)
        self.attention.set_mask_k(mask)

    def forward(self, inputs):
        x = inputs
        res = x
        x = self.attention(x, x, x)
        x = self.lnorm1(res + self.dropout(x))
        res = x
        x = self.fc(x)
        x = self.lnorm2(res + self.dropout(x))

        return x


class DecoderBlock(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, inner_linear=1024, dropout=0):

        super(DecoderBlock, self).__init__()
        self.lnorm1 = LayerNorm1d(hidden_size)
        self.lnorm2 = LayerNorm1d(hidden_size)
        self.lnorm3 = LayerNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout=dropout, causal=False)
        self.masked_attention = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout=dropout, causal=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, inner_linear),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(inner_linear, hidden_size))

    def set_mask(self, mask, context_mask=None):
        if context_mask is not None:
            self.attention.set_mask_k(context_mask)
        self.masked_attention.set_mask_q(mask)
        self.masked_attention.set_mask_k(mask)

    def forward(self, inputs, context):
        x = inputs
        res = x
        x = self.masked_attention(x, x, x)
        x = self.lnorm1(res + self.dropout(x))
        res = x
        x = self.attention(x, context, context)
        x = self.lnorm2(res + self.dropout(x))
        res = x
        x = self.fc(x)
        x = self.lnorm3(res + self.dropout(x))

        return x


class TransformerAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None,
                 num_layers=6, num_heads=8, inner_linear=1024,
                 mask_symbol=PAD, dropout=0):

        super(TransformerAttentionEncoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(hidden_size, num_heads, inner_linear, dropout)
                                     for _ in range(num_layers)
                                     ])

    def forward(self, inputs, hidden=None):
        if self.mask_symbol is not None:
            padding_mask = inputs.data.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs) * self.scale_embedding
        x = x + Variable(positional_embedding(x), requires_grad=False)
        x = self.dropout(x)

        for block in self.blocks:
            block.set_mask(padding_mask)
            x = block(x)

        return x, padding_mask


class TransformerAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None,
                 num_layers=6, num_heads=8, dropout=0, inner_linear=1024,
                 mask_symbol=PAD, tie_embedding=True):

        super(TransformerAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(hidden_size, num_heads, inner_linear, dropout)
                                     for _ in range(num_layers)
                                     ])
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.embedder.weight = self.classifier.weight

    def forward(self, inputs, context):
        context_enc, context_mask = context
        if self.mask_symbol is not None:
            padding_mask = inputs.data.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs) * self.scale_embedding
        x = x + Variable(positional_embedding(x), requires_grad=False)
        x = self.dropout(x)

        for block in self.blocks:
            block.set_mask(padding_mask, context_mask)
            x = block(x, context_enc)
        x = self.classifier(x)
        return x, context


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6,
                 num_heads=8, inner_linear=2048, dropout=0.1, tie_embedding=True,
                 encoder=None, decoder=None):
        super(Transformer, self).__init__(batch_first=True)
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('num_layers', num_layers)
        encoder.setdefault('num_heads', num_heads)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('dropout', dropout)
        encoder.setdefault('inner_linear', inner_linear)

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('num_heads', num_heads)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('inner_linear', inner_linear)

        self.encoder = TransformerAttentionEncoder(**encoder)
        self.decoder = TransformerAttentionDecoder(**decoder)

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.classifier.weight

    def generate(self, input_list, state_list, k=1, feed_all_timesteps=True, get_attention=False):
        # TODO cache computation, not inputs
        return super(Transformer, self).generate(input_list, state_list, k=k,
                                                 feed_all_timesteps=feed_all_timesteps,
                                                 get_attention=get_attention)

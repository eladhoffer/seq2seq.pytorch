import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .modules import LayerNorm1d
from .attention import MultiHeadAttention
from .seq2seq import Seq2Seq
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
        self.attention.set_mask(mask)

    def forward(self, inputs):
        x = inputs
        res = x
        x = self.attention(x, x, x)
        x = self.lnorm1(res + self.dropout(x))
        res = x
        x = x.view(-1, x.size(2))
        x = self.fc(x)
        x = x.view(res.size(0), res.size(1), res.size(2))
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

    def set_mask(self, mask):
        self.attention.set_mask(mask)
        self.masked_attention.set_mask(mask)

    def forward(self, inputs, context):
        x = inputs
        res = x
        x = self.masked_attention(x, x, x)
        x = self.lnorm1(res + self.dropout(x))
        res = x
        x = self.attention(x, context, context)
        x = self.lnorm2(res + self.dropout(x))
        res = x
        x = x.view(-1, x.size(2))
        x = self.fc(x)
        x = x.view(res.size(0), res.size(1), res.size(2))
        x = self.lnorm3(res + self.dropout(x))

        return x


class TransformerAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, inner_linear=1024, mask_symbol=0, dropout=0):

        super(TransformerAttentionEncoder, self).__init__()
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(hidden_size, num_heads, inner_linear, dropout)
                                     for _ in range(num_layers)
                                     ])

    def forward(self, inputs, hidden=None):
        if self.mask_symbol:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs) * self.scale_embedding
        x = x + Variable(positional_embedding(x), requires_grad=False)
        x = self.dropout(x)

        for block in self.blocks:
            block.set_mask(padding_mask)
            x = block(x)

        return x


class TransformerAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, dropout=0, inner_linear=1024, tie_embedding=True):

        super(TransformerAttentionDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(hidden_size, num_heads, inner_linear, dropout)
                                     for _ in range(num_layers)
                                     ])
        self.classifier = nn.Linear(hidden_size, vocab_size)
        if tie_embedding:
            self.embedder.weight = self.classifier.weight

    def forward(self, inputs, context):
        x = self.embedder(inputs) * self.scale_embedding
        x = x + Variable(positional_embedding(x), requires_grad=False)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, context)
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)
        return x, context


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, inner_linear=2048, dropout=0.1, tie_embedding=True):
        super(Transformer, self).__init__(batch_first=True)
        self.encoder = TransformerAttentionEncoder(vocab_size, hidden_size=hidden_size,
                                                   num_layers=num_layers, num_heads=num_heads, inner_linear=inner_linear,
                                                   dropout=dropout)
        self.decoder = TransformerAttentionDecoder(vocab_size, hidden_size=hidden_size,
                                                   num_layers=num_layers, num_heads=num_heads, inner_linear=inner_linear,
                                                   dropout=dropout, tie_embedding=tie_embedding)

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.classifier.weight

    def generate(self, input_list, state_list, k=1, feed_all_timesteps=True, get_attention=False):
        # TODO cache computation, not inputs
        return super(Transformer, self).generate(input_list, state_list, k=k,
                                                 feed_all_timesteps=feed_all_timesteps,
                                                 get_attention=get_attention)
        # return self.decode(inputs, context)
        # if hasattr(self, 'cache') and self.cache is not None:
        #     self.cache = self.cache.expand(inputs.size(0), self.cache.size(1))
        #     inputs = torch.cat([self.cache, inputs], 1)
        # self.cache = inputs.clone()
        # return self.decode(self.cache, context)

    def clear_state(self):
        self.cache = None

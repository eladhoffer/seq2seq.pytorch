import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .modules import LayerNorm1d
from .attention import MultiHeadAttention
from .seq2seq import Seq2Seq


def pos_embedding(x, scale=1000):
    b, t, dim = list(x.size())
    assert (dim % 2 == 0)
    range_vec = torch.arange(0, dim, 2).float() / dim
    range_vec = range_vec.unsqueeze(0).expand(t, dim // 2)
    pos_vec = torch.arange(0, t).float().unsqueeze(1).expand(t, dim // 2)
    if x.is_cuda:
        pos_vec = pos_vec.cuda()
        range_vec = range_vec.cuda()
    mat = pos_vec * torch.pow(float(scale), range_vec)
    return torch.cat([mat.sin(), mat.cos()], 1).unsqueeze(0).expand(b, t, dim)


class EncoderBlock(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, inner_linear=1024, dropout=0):

        super(EncoderBlock, self).__init__()
        self.lnorm = LayerNorm1d(hidden_size)
        self.attention = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, causal=False)
        self.fc = nn.Sequential(nn.Linear(hidden_size, inner_linear),
                                nn.ReLU(inplace=True),
                                nn.Linear(inner_linear, hidden_size))

    def forward(self, inputs):
        x = inputs
        res = x
        x = self.attention(x, x, x)
        x = res + x
        x = self.lnorm(x)
        res = x
        x = x.view(-1, x.size(2))
        x = self.fc(x)
        x = x.view(res.size(0), res.size(1), res.size(2))
        x = res + x
        x = self.lnorm(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, inner_linear=1024, dropout=0):

        super(DecoderBlock, self).__init__()
        self.lnorm = LayerNorm1d(hidden_size)
        self.attention = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, causal=False)
        self.masked_attention = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, causal=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, inner_linear),
                                nn.ReLU(inplace=True),
                                nn.Linear(inner_linear, hidden_size))

    def forward(self, inputs, context):
        x = inputs
        res = x
        x = self.masked_attention(x, x, x)
        x = res + x
        x = self.lnorm(x)
        res = x
        x = self.attention(x, context, context)
        x = res + x
        x = self.lnorm(x)
        res = x
        x = x.view(-1, x.size(2))
        x = self.fc(x)
        x = x.view(res.size(0), res.size(1), res.size(2))
        x = res + x
        x = self.lnorm(x)

        return x


class TransformerAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, inner_linear=1024, dropout=0):

        super(TransformerAttentionEncoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([EncoderBlock(hidden_size, num_heads, inner_linear, dropout)
                                     for _ in range(num_layers)
                                     ])

    def forward(self, inputs, hidden=None):
        x = self.embedder(inputs)
        x = x + Variable(pos_embedding(x), requires_grad=False)

        for block in self.blocks:
            x = block(x)

        return x


class TransformerAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, dropout=0, inner_linear=1024, tie_embedding=True):

        super(TransformerAttentionDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(hidden_size, num_heads, inner_linear, dropout)
                                     for _ in range(num_layers)
                                     ])
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, context):
        x = self.embedder(inputs)
        x = x + Variable(pos_embedding(x), requires_grad=False)

        for block in self.blocks:
            x = block(x, context)
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)
        return x, context


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, dropout=0, tie_embedding=True):
        super(Transformer, self).__init__()
        self.encoder = TransformerAttentionEncoder(vocab_size, hidden_size=hidden_size,
                                                   num_layers=num_layers, num_heads=num_heads, dropout=dropout)
        self.decoder = TransformerAttentionDecoder(vocab_size, hidden_size=hidden_size,
                                                   num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                                                   tie_embedding=tie_embedding)

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

    def generate(self, inputs, context):
        # TODO cache computation, not inputs
        cached_inputs = inputs
        if hasattr(self, 'cache') and self.cache is not None:
            self.cache = self.cache.expand(inputs.size(0), self.cache.size(1))
            cached_inputs = torch.cat([self.cache, inputs], 1)
        self.cache = inputs.clone()
        output, _ = self.decode(cached_inputs, context)
        return output, context

    def clear_state(self):
        self.cache = None

# # test:
# model = Transformer(1000, 128, 3)
# x1 = torch.autograd.Variable(torch.rand(32, 16).long().fill_(2))
# x2 = torch.autograd.Variable(torch.rand(32, 16).long().fill_(2))
# y = model(x1, x2)

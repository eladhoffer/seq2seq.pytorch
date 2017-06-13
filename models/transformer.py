import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import LayerNorm1d
from .attention import MultiHeadAttention
from .seq2seq import Seq2Seq

class AttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, dropout=0):

        super(AttentionEncoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.lnorm = LayerNorm1d(hidden_size)
        self.attention_layers = nn.ModuleList(([MultiHeadAttention(hidden_size, hidden_size, num_heads, causal=False)
                                                for _ in range(num_layers)
                                                ]))
        self.linear_layers = nn.ModuleList(([nn.Linear(hidden_size, hidden_size)
                                             for _ in range(num_layers)
                                             ]))

    def forward(self, inputs, hidden=None):
        x = self.embedder(inputs)
        for i in range(len(self.attention_layers)):
            res = x
            x = self.attention_layers[i](x, x, x)
            x = res + x
            # x = self.lnorm(x)
            res = x
            x = x.view(-1, x.size(2))
            x = self.linear_layers[i](x)
            x = x.view(res.size(0), res.size(1), res.size(2))
            x = res + x
            # x = self.lnorm(x)

        return x


class AttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, dropout=0, tie_embedding=True):

        super(AttentionDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.lnorm = LayerNorm1d(hidden_size)
        self.masked_attention_layers = nn.ModuleList(([MultiHeadAttention(hidden_size, hidden_size, num_heads, causal=True)
                                                       for _ in range(num_layers)
                                                       ]))
        self.attention_layers = nn.ModuleList(([MultiHeadAttention(hidden_size, hidden_size, num_heads, causal=False)
                                                for _ in range(num_layers)
                                                ]))
        self.linear_layers = nn.ModuleList(([nn.Linear(hidden_size, hidden_size)
                                             for _ in range(num_layers)
                                             ]))
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, context):
        x = self.embedder(inputs)
        for i in range(len(self.attention_layers)):
            res = x
            x = self.masked_attention_layers[i](x, x, x)
            x = res + x
            # x = self.lnorm(x)
            res = x
            x = self.attention_layers[i](x, context, context)
            x = res + x
            # x = self.lnorm(x)
            res = x
            x = x.view(-1, x.size(2))
            x = self.linear_layers[i](x)
            x = x.view(res.size(0), res.size(1), res.size(2))
            x = res + x
            # x = self.lnorm(x)
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)
        return x, context


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, num_layers=6, dropout=0, tie_enc_dec_embedding=True):
        super(Transformer, self).__init__()
        self.encoder = AttentionEncoder(vocab_size, hidden_size=hidden_size,
                                        num_layers=num_layers, dropout=dropout)
        self.decoder = AttentionDecoder(vocab_size, hidden_size=hidden_size,
                                        num_layers=num_layers, dropout=dropout,
                                        tie_embedding=tie_enc_dec_embedding)

        if tie_enc_dec_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

#test:
# model = Transformer(1000, 128, 3)
# x1 = torch.autograd.Variable(torch.rand(32, 16).long().fill_(2))
# x2 = torch.autograd.Variable(torch.rand(32, 16).long().fill_(2))
# y = model(x1, x2)

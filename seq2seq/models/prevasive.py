import torch
import torch.nn as nn
import math
from .seq2seq_base import Seq2Seq
from seq2seq.tools.config import PAD
from .modules.state import State
from .modules.transformer_blocks import positional_embedding
from .modules.prevasive_resnet import ResNet
from .modules.prevasive_densenet import DenseNet
from .transformer import TransformerAttentionEncoder

def merge_time(x1, x2, x3=None, mode='cat'):
    B1, C1, T1 = x1.shape
    B2, C2, T2 = x2.shape
    assert B1 == B2
    if mode == 'cat':
        shape1 = list(x1.shape)
        shape1.insert(3, T2)
        shape2 = list(x2.shape)
        shape2.insert(2, T1)

        return torch.cat((x1.unsqueeze(3).expand(shape1),
                          x2.unsqueeze(2).expand(shape2)), dim=1)
    elif mode == 'sum':
        assert C1 == C2
        return x1.unsqueeze(3) + x2.unsqueeze(2)
    elif mode == 'film':
        assert x3 is not None
        assert C1 == C2
        assert x3.shape == x2.shape
        return (x1.unsqueeze(3) * x2.unsqueeze(2)) + x3.unsqueeze(2)


class PrevasiveEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None,
                 encoding='embedding', mask_symbol=PAD,  dropout=0):

        super(PrevasiveEncoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.hidden_size = hidden_size
        self.batch_first = True
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.hidden_size = embedding_size
        self.encoding = encoding

    def forward(self, inputs, hidden=None):
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        x.add_(positional_embedding(x))
        x = self.dropout(x)

        return State(outputs=x, mask=padding_mask, batch_first=True)


class PrevasiveDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, context_size=None,
                 dropout=0,  convnet='resnet', merge_time='cat', stateful=False, mask_symbol=PAD, tie_embedding=True):

        super(PrevasiveDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        context_size = context_size or hidden_size

        self.batch_first = True
        self.mask_symbol = mask_symbol
        self.embedder = nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.merge_time = merge_time
        if self.merge_time == 'cat':
            embed_input = embedding_size + context_size
        else:
            embed_input = embedding_size
        if convnet == 'resnet':
            self.main_block = ResNet(embed_input, output_size=embedding_size)
        elif convnet == 'densenet':
            self.main_block = DenseNet(
                embed_input, output_size=embedding_size, block_config=(6, 6, 6, 8), growth_rate=40)
        self.pool = nn.AdaptiveMaxPool2d((1, None))
        self.classifier = nn.Linear(embedding_size, vocab_size)
        if tie_embedding:
            self.embedder.weight = self.classifier.weight

    def forward(self, inputs, state, get_attention=False):
        context = state.context  # context has outputs and mask
        time_step = 0
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None

        x = self.embedder(inputs).mul_(self.scale_embedding)
        x.add_(positional_embedding(x, offset=time_step))
        x = self.dropout(x)
        x = x.transpose(1, 2)

        y = context.outputs.transpose(1, 2)

        # x is B x C2 x T2
        # y is B x C1 x T1
        x = merge_time(y, x, mode=self.merge_time)  # B x (C1+C2) x T1 x T2
        x = self.main_block(x)  # B x Cout x T1 x T2
        x = self.pool(x)  # B x Cout x 1 x T2
        x = x.squeeze(2).transpose(1, 2)  # B x T2 x Cout
        x = self.classifier(x)

        return x, state


class Prevasive(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6, dropout=0.1, tie_embedding=True,
                 encoder=None, decoder=None, merge_time='cat'):
        super(Prevasive, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder_type = encoder.pop('type', 'simple')
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('dropout', dropout)

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('merge_time', merge_time)
        decoder.setdefault('dropout', dropout)

        self.batch_first = True
        if encoder_type == 'simple':
            self.encoder = PrevasiveEncoder(**encoder)
        elif encoder_type == 'transformer':
            self.encoder = TransformerAttentionEncoder(**encoder)
        decoder['context_size'] = self.encoder.hidden_size
        self.decoder = PrevasiveDecoder(**decoder)

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.classifier.weight

    def _decode_step(self, input_list, state_list, k=1,
                     feed_all_timesteps=True,
                     remove_unknown=False,
                     get_attention=False):
        return super(Prevasive, self)._decode_step(input_list, state_list=state_list, k=k,
                                                   feed_all_timesteps=feed_all_timesteps,
                                                   remove_unknown=remove_unknown,
                                                   get_attention=get_attention)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .seq2seq_base import Seq2Seq
from .recurrent import RecurrentAttentionDecoder, RecurrentEncoder
from .modules.vision_encoders import AlexNetEncoder, ResNetEncoder, DenseNetEncoder, VGGEncoder, SqueezeNetEncoder
from .modules.state import State


class Img2Seq(Seq2Seq):

    def __init__(self, vocab_size, encoder=None, decoder=None):
        super(Img2Seq, self).__init__()
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {'model': 'resnet50'}
        decoder = decoder or {}
        decoder.setdefault('hidden_size', 128)
        decoder.setdefault('embedding_size', decoder['hidden_size'])
        decoder.setdefault('num_layers', 1)
        decoder.setdefault('bias', True)
        decoder.setdefault('tie_embedding', False)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('dropout', 0)
        decoder.setdefault('residual', False)
        decoder.setdefault('batch_first', False)

        if 'resnet' in encoder['model']:
            self.encoder = ResNetEncoder(**encoder)
        elif 'densenet' in encoder['model']:
            self.encoder = DenseNetEncoder(**encoder)
        elif 'vgg' in encoder['model']:
            self.encoder = VGGEncoder(**encoder)
        elif 'alexnet' in encoder['model']:
            self.encoder = AlexNetEncoder(**encoder)
        elif 'squeezenet' in encoder['model']:
            self.encoder = SqueezeNetEncoder(**encoder)
        decoder['context_size'] = self.encoder.hidden_size

        self.decoder = RecurrentAttentionDecoder(**decoder)

    def encode(self, x, hidden=None, devices=None):
        x = x.squeeze(0)
        x = self.encoder(x)

        return State(outputs=x, batch_first=True)

    def load_state_dict(self, state_dict):
        try:
            super(Img2Seq, self).load_state_dict(state_dict)
        except:
            finetune = self.encoder.finetune
            self.encoder.finetune = False
            super(Img2Seq, self).load_state_dict(state_dict)
            self.encoder.finetune = finetune

    def bridge(self, context):
        B, C, H, W = list(context.outputs.size())
        context.outputs = context.outputs.view(B, C, H * W)
        context.outputs = context.outputs.transpose(1, 2)
        # B x H*W x C
        if not self.decoder.batch_first:  # H*W x B x C
            context.outputs = context.outputs.transpose(0, 1)
        context.batch_first = self.decoder.batch_first
        return State(context=context, batch_first=self.decoder.batch_first)

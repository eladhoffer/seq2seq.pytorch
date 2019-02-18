import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .seq2seq_base import Seq2Seq
from .recurrent import RecurrentDecoder, RecurrentAttentionDecoder, RecurrentEncoder
from .transformer import TransformerAttentionDecoder
from .modules.vision_encoders import AlexNetEncoder, ResNetEncoder, DenseNetEncoder, VGGEncoder, SqueezeNetEncoder
from .modules.state import State


class Img2Seq(Seq2Seq):

    def __init__(self, vocab_size, encoder=None, decoder=None, transfer_hidden=False):
        super(Img2Seq, self).__init__()
        self.transfer_hidden = transfer_hidden
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        encoder = deepcopy(encoder)
        model_name = encoder.get('model', 'resnet50')
        encoder.setdefault('context_transform', None)
        encoder.setdefault('spatial_context', True)

        decoder = decoder or {}
        decoder = deepcopy(decoder)
        decoder_type = decoder.pop('type', 'recurrent_attention')

        if 'resnet' in model_name:
            self.encoder = ResNetEncoder(**encoder)
        elif 'densenet' in model_name:
            self.encoder = DenseNetEncoder(**encoder)
        elif 'vgg' in model_name:
            self.encoder = VGGEncoder(**encoder)
        elif 'alexnet' in model_name:
            self.encoder = AlexNetEncoder(**encoder)
        elif 'squeezenet' in model_name:
            self.encoder = SqueezeNetEncoder(**encoder)

        if decoder_type == 'recurrent_attention':
            decoder['context_size'] = self.encoder.context_size
            self.decoder = RecurrentAttentionDecoder(**decoder)
        elif decoder_type == 'recurrent':
            decoder['context_size'] = self.encoder.context_size
            self.decoder = RecurrentDecoder(**decoder)
        elif decoder_type == 'transformer':
            decoder['hidden_size'] = self.encoder.context_size
            self.decoder = TransformerAttentionDecoder(**decoder)
        self.batch_first = getattr(self.decoder, 'batch_first', False)

    def encode(self, x, hidden=None, device_ids=None):
        x = x.squeeze(0)
        x = self.encoder(x)
        return State(outputs=x, batch_first=True)

    def load_state_dict(self, state_dict, **kwargs):
        try:
            super(Img2Seq, self).load_state_dict(state_dict, **kwargs)
        except:
            finetune = self.encoder.finetune
            self.encoder.finetune = False
            super(Img2Seq, self).load_state_dict(state_dict, **kwargs)
            self.encoder.finetune = finetune

    def bridge(self, context):
        if context.outputs.dim() > 2:  # spatial output -- translate to time
            B, C, H, W = list(context.outputs.size())
            context.outputs = context.outputs.view(B, C, H * W)
            context.outputs = context.outputs.transpose(1, 2)
            # B x H*W x C
            if not self.decoder.batch_first:  # H*W x B x C
                context.outputs = context.outputs.transpose(0, 1)
        if self.transfer_hidden:
            hidden = context.outputs
            if hasattr(self.decoder, 'rnn'):
                num_layers = self.decoder.rnn.num_layers
                hidden = hidden.unsqueeze(0)
                hidden = hidden.expand(num_layers, *list((hidden.size())[1:]))
                if getattr(self.decoder.rnn, 'mode') == 'LSTM':
                    hidden = (hidden, hidden)
        else:
            hidden = None
        context.batch_first = self.decoder.batch_first
        return State(hidden=hidden, context=context, batch_first=self.decoder.batch_first)

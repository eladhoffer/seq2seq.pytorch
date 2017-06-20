import torch
import torch.nn as nn
from torch.autograd import Variable
from .seq2seq import Seq2Seq
from .recurrent import RecurrentAttentionDecoder, RecurrentEncoder
from torchvision.models import resnet


# models = dict(**resnet.__dict__, **vgg.__dict__)


class ResNetCaptionGenerator(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=256, model_name='resnet50',
                 num_layers=2, bias=True, dropout=0, tie_embedding=False):
        super(ResNetCaptionGenerator, self).__init__()
        self.encoder = resnet.__dict__[model_name](pretrained=True)
        self.decoder = RecurrentAttentionDecoder(vocab_size, hidden_size=hidden_size,
                                                 tie_embedding=tie_embedding, context_size=2048,
                                                 num_layers=num_layers, bias=bias, dropout=dropout)

    def encode(self, x, hidden=None, devices=None):
        x = x.squeeze(0)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        return x, None

    def bridge(self, context):
        context, hidden = context
        B, C, H, W = list(context.size())
        context = context.view(B, C, H * W) \
            .transpose(0, 1) \
            .transpose(0, 2)  # H*W x B x C
        return (context, hidden)

    def generate(self, inputs, context, get_attention=False):
        return self.decoder(inputs, context, get_attention=get_attention)

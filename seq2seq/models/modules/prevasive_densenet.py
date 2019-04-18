import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .conv import MaskedConv2d, TimeNorm2d


def init_model(model):
    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=True)
        # self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = MaskedConv2d(bn_size * growth_rate, growth_rate,
                                  kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate

    def forward(self, inputs):
        # x = self.norm1(inputs)
        x = self.relu(inputs)
        x = self.conv1(x)
        # x = self.norm2(x)
        x = self.relu(x)
        new_features, _ = self.conv2(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([inputs, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=True))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=(2,1), stride=(2, 1)))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, input_size, output_size, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNet, self).__init__()
        num_init_features = input_size
        # self.conv0 = MaskedConv2d(input_size, num_init_features,
                                #   kernel_size=7, stride=(2, 1), padding=3, bias=True)
        # self.norm0 = nn.BatchNorm2d(num_init_features)
        # self.relu0 = nn.ReLU(inplace=True)
        # self.pool0 = nn.MaxPool2d(kernel_size=(3, 1),
                                #   stride=(2, 1), padding=(1, 0))
        self.features = nn.Sequential()
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.conv2 = nn.Conv2d(num_features, output_size,
                               kernel_size=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(inplace=True)
        # Linear layer
        init_model(self)

    def forward(self, x):
        # x, _ = self.conv0(x)
        # x = self.norm0(x)
        # x = self.relu(x)
        # x = self.pool0(x)
        features = self.features(x)
        out = F.relu(features)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)
        return out

import torch
import torch.nn as nn
import math
from .conv import MaskedConv2d, TimeNorm2d


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=(3, 3), stride=1, expansion=4, downsample=None, groups=1, residual_block=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(
            planes, planes, 3, padding=1, stride=(stride, 1), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x, cache=None):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out, cache = self.conv2(out, cache)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)
        out += residual
        out = self.relu(out)

        return out


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, MaskedConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)


class ResNet(nn.Module):

    def __init__(self, input_size, output_size, inplanes=128, strided=True,
                 block=Bottleneck, residual_block=None, layers=[1, 1, 1, 1],
                 width=[128, 128, 128, 128], expansion=2, groups=[1, 1, 1, 1]):
        super(ResNet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = MaskedConv2d(input_size, self.inplanes, kernel_size=7, stride=(2, 1), padding=3,
                                  bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        for i in range(len(layers)):
            if strided:
                stride = 1 if i == 0 else 2
            else:
                stride = 1
            setattr(self, 'layer%s' % str(i + 1),
                    self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion,
                                     stride=stride, residual_block=residual_block, groups=groups[i]))  # 1 if i == 0 else 2

        self.conv2 = nn.Conv2d(self.inplanes, output_size,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(inplace=True)
        init_model(self)

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes,
                          kernel_size=1, stride=(stride, 1), bias=True),
                nn.BatchNorm2d(planes * expansion),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block))

        return nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

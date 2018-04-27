import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.conv import MaskedConv1d

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, interm_channels=None, out_channels=None, kernel_size=3, dilation=1, causal=True):
        super(ResidualBlock, self).__init__()
        out_channels = out_channels or in_channels
        interm_channels = interm_channels or in_channels // 2
        self.layernorm1 = nn.LayerNorm(in_channels)
        self.layernorm2 = nn.LayerNorm(interm_channels)
        self.layernorm3 = nn.LayerNorm(interm_channels)
        self.conv1 = nn.Conv1d(in_channels, interm_channels, 1)
        self.conv2 = MaskedConv1d(
            interm_channels, interm_channels, kernel_size, dilation=dilation, causal=causal)
        self.conv3 = nn.Conv1d(interm_channels, out_channels, 1)
        self.relu = nn.ReLU(True)

    def forward(self, inputs):
        out = self.layernorm1(inputs)
        out = self.conv1(self.relu(out))
        out = self.layernorm2(out)
        out = self.conv2(self.relu(out))
        out = self.layernorm3(out)
        out = self.conv3(self.relu(out))
        out += inputs
        return out


class ByteNet(nn.Sequential):

    def __init__(self, num_channels=512, num_sets=6, dilation_rates=[1, 2, 4, 8, 16], kernel_size=3, block=ResidualBlock, causal=True):
        super(ByteNet, self).__init__()
        for s in range(num_sets):
            for r in dilation_rates:
                self.add_module('block%s_%s' % (s, r),
                                block(num_channels, kernel_size=kernel_size, dilation=r, causal=causal))

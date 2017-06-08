import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2

        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)

    def forward(self, inputs):
        output = F.conv1d(inputs, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output[:, :, :inputs.size(2)]


class LayerNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-05, affine=True, bias_init=0):
        super(LayerNorm1d, self).__init__()
        self.eps = eps
        self.bias_init = bias_init
        self.num_features = num_features
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.)
            self.bias.data.fill_(self.bias_init)

    def forward(self, inputs):
        mean = inputs.mean(1).expand_as(inputs)
        input_centered = inputs - mean
        std = (input_centered ** 2).mean(1).sqrt().expand_as(inputs)
        output = input_centered / (std + self.eps)

        if self.affine:
            w = self.weight.view(1, -1, 1).expand_as(output)
            b = self.bias.view(1, -1, 1).expand_as(output)
            output = output * w + b
        return output


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, interm_channels=None, out_channels=None, kernel_size=3, dilation=1, causal=True):
        super(ResidualBlock, self).__init__()
        out_channels = out_channels or in_channels
        interm_channels = interm_channels or in_channels // 2
        self.layernorm1 = LayerNorm1d(in_channels)
        self.layernorm2 = LayerNorm1d(interm_channels)
        self.layernorm3 = LayerNorm1d(interm_channels)
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


#
# class MU(nn.Module):
#
#     def __init__():
#         super(MU, self).__init__()
#
#     def forward(inputs):
#         hx, cx = hidden
#         gates = F.linear(inputs, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
#         ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#
#         ingate = F.sigmoid(ingate)
#         forgetgate = F.sigmoid(forgetgate)
#         cellgate = F.tanh(cellgate)
#         outgate = F.sigmoid(outgate)
#
#         cy = (forgetgate * cx) + (ingate * cellgate)
#         hy = outgate * F.tanh(cy)

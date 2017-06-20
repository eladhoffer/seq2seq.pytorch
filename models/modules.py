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
        output = super(MaskedConv1d, self).forward(inputs)
        return output[:, :, :inputs.size(2)]


class GatedConv1d(MaskedConv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        super(GatedConv1d, self).__init__(in_channels, 2 * out_channels,
                                          kernel_size, dilation, groups, bias, causal)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        output = super(GatedConv1d, self).forward(inputs)
        mask, output = output.chunk(2, 1)
        mask = self.sigmoid(mask)

        return output * mask


class LayerNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-6, affine=True):
        super(LayerNorm1d, self).__init__()
        self.eps = eps
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
            self.bias.data.fill_(0.)

    def forward(self, inputs):
        b, t, _ = list(inputs.size())
        mean = inputs.mean(2).view(b, t, 1).expand_as(inputs)
        input_centered = inputs - mean
        std = input_centered.pow(2).mean(2).add(self.eps).sqrt()
        output = input_centered / std.view(b, t, 1).expand_as(inputs)

        if self.affine:
            w = self.weight.view(1, 1, -1).expand_as(output)
            b = self.bias.view(1, 1, -1).expand_as(output)
            output = output * w + b
        return output

# class SimpleLSTMCell(nn.Module):
#
#     def __init__(self, input_size, hidden_size, bias=True):
#         super(SimpleLSTMCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
#         self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, hidden):
#         hx, cx = hidden
#         gates = self.ih(input)
#         gates += self.hh(hx)
#
#         ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#
#         ingate = self.sigmoid(ingate)
#         forgetgate = self.sigmoid(forgetgate)
#         cellgate = self.tanh(cellgate)
#         outgate = self.sigmoid(outgate)
#
#         cy = (forgetgate * cx) + (ingate * cellgate)
#         hy = outgate * self.tanh(cy)
#
#         return hy, cy
#

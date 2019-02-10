import torch
import torch.nn as nn
import torch.nn.functional as F

def _sum_tensor_scalar(tensor, scalar, expand_size):
    if scalar is not None:
        scalar = scalar.expand(expand_size).contiguous()
    else:
        return tensor
    if tensor is None:
        return scalar
    return tensor + scalar


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, groups=1,
                 multiplier=False, pre_bias=False, post_bias=False):
        if in_features % groups != 0:
            raise ValueError('in_features must be divisible by groups')
        if out_features % groups != 0:
            raise ValueError('out_features must be divisible by groups')
        self.groups = groups
        super(Linear, self).__init__(in_features,
                                     out_features // self.groups, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.full((out_features,), 0))
        else:
            self.register_parameter('bias', None)

        if pre_bias:
            self.pre_bias = nn.Parameter(torch.tensor([0.]))
        else:
            self.register_parameter('pre_bias', None)
        if post_bias:
            self.post_bias = nn.Parameter(torch.tensor([0.]))
        else:
            self.register_parameter('post_bias', None)
        if multiplier:
            self.multiplier = nn.Parameter(torch.tensor([1.]))
        else:
            self.register_parameter('multiplier', None)

    def forward(self, x):
        if self.pre_bias is not None:
            x = x + self.pre_bias
        weight = self.weight if self.multiplier is None\
            else self.weight * self.multiplier
        bias = _sum_tensor_scalar(self.bias, self.post_bias, self.out_features)
        if self.groups == 1:
            out = F.linear(x, weight, bias)
        else:
            x_g = x.chunk(self.groups, dim=-1)
            w_g = weight.chunk(self.groups, dim=-1)
            out = torch.cat([F.linear(x_g[i], w_g[i])
                             for i in range(self.groups)], -1)
            if bias is not None:
                out += bias
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, groups=1):
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

    def forward(self, x):
        if self.groups == 1:
            out = super(Linear, self).forward(x)
        else:
            x_g = x.chunk(self.groups, dim=-1)
            w_g = self.weight.chunk(self.groups, dim=-1)
            out = torch.cat([F.linear(x_g[i], w_g[i])
                             for i in range(self.groups)], -1)
            if self.bias is not None:
                out += self.bias
        return out

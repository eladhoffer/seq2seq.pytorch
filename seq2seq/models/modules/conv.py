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


class MaskedConv2d(nn.Conv2d):
    """masked over 3rd dimension (2nd spatial dimension)"""

    def __init__(self, *kargs, **kwargs):
        super(MaskedConv2d, self).__init__(*kargs, **kwargs)
        self.masked_dim = 1

        def pad_needed(causal, size, stride, pad, dilation):
            if not causal:
                return 0
            else:
                return (size - 1) * dilation

        add_padding = (pad_needed(self.masked_dim == 0, self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0]),
                       pad_needed(self.masked_dim == 1, self.kernel_size[1], self.stride[1], self.padding[1], self.dilation[1]))
        self.padding = (self.padding[0] + add_padding[0] // 2,
                        self.padding[1] + add_padding[1] // 2)
        self.remove_padding = add_padding

    def forward(self, inputs):
        output = super(MaskedConv2d, self).forward(inputs)
        return output[:, :, :output.size(2)-self.remove_padding[0], :output.size(3)-self.remove_padding[1]]


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


class TimeNorm2d(nn.InstanceNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(TimeNorm2d, self).__init__(num_features,
                                         eps, momentum, affine, track_running_stats)

    def forward(self, inputs):
        B, C, T1, T2 = inputs.shape
        x = inputs.transpose(0, 3)  # T2 x C x T1 x B
        # x = x.contiguous().view(B*T1, C, 1, T2)

        y = super(TimeNorm2d, self).forward(x)
        y = y.transpose(0, 3)
        return y


if __name__ == "__main__":
    x = torch.rand(16, 32, 30, 52).fill_(1)
    m = MaskedConv2d(32, 64, 3, padding=1, bias=False)
    m.weight.detach().fill_(1)
    y = m(x)
    print(x.shape)
    print(y.shape)

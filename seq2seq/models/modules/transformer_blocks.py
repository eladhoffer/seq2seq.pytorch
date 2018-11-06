import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .weight_norm import weight_norm as wn
from .linear import Linear
from .recurrent import Recurrent


def positional_embedding(x, min_timescale=1.0, max_timescale=1.0e4, offset=0):
    batch, length, channels = list(x.size())
    assert (channels % 2 == 0)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (float(num_timescales) - 1.))
    position = torch.arange(offset, offset + length,
                            device=x.device, dtype=torch.float)
    inv_timescales = torch.arange(0, num_timescales,
                                  device=x.device, dtype=torch.float)

    inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    # scaled time is now length x num_timescales
    # length x channels
    signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)
    return signal.unsqueeze(0).expand(batch, length, channels)


class AverageNetwork(nn.Module):
    """ Average Attention Network Block from https://arxiv.org/abs/1805.00631 
    """

    def __init__(self, input_size, inner_linear, inner_groups=1, layer_norm=True, weight_norm=False, dropout=0, batch_first=True):
        super(AverageNetwork, self).__init__()
        wn_func = wn if weight_norm else lambda x: x
        self.input_size = input_size
        self.time_step = 0
        self.batch_dim, self.time_dim = (0, 1) if batch_first else (1, 0)
        self.gates = nn.Sequential(
            wn_func(nn.Linear(2 * input_size, 2 * input_size)),
            nn.Sigmoid()
        )
        if layer_norm:
            self.lnorm = nn.LayerNorm(input_size)
        self.fc = nn.Sequential(wn_func(Linear(input_size, inner_linear, groups=inner_groups)),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                wn_func(Linear(inner_linear, input_size, groups=inner_groups)))

    def forward(self, x, state=None):
        if state is None:
            self.time_step = 0
        num_steps = torch.arange(
            self.time_step + 1, x.size(self.time_dim) + self.time_step + 1,
            device=x.device, dtype=x.dtype).view(-1, 1)

        if state is None:
            avg_attn = x.cumsum(self.time_dim) / num_steps
        else:
            past_num_steps = self.time_step
            avg_attn = ((self.time_step * state.unsqueeze(self.time_dim)) +
                        x.cumsum(self.time_dim)) / num_steps

        state = avg_attn.select(self.time_dim, -1)
        g = self.fc(avg_attn)

        # gating
        gate_values = self.gates(torch.cat((x, g), dim=-1))
        input_gate, forget_gate = gate_values.chunk(2, dim=-1)
        output = (input_gate + 1) * x + forget_gate * g  # gate and residual
        if hasattr(self, 'lnorm'):
            output = self.lnorm(output)

        self.time_step += x.size(self.time_dim)
        return output, state


class EncoderBlock(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, inner_linear=2048, inner_groups=1,
                 layer_norm=True, weight_norm=False, dropout=0):

        super(EncoderBlock, self).__init__()
        wn_func = wn if weight_norm else lambda x: x
        if layer_norm:
            self.lnorm1 = nn.LayerNorm(hidden_size)
            self.lnorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout=dropout, causal=False, weight_norm=weight_norm)
        self.fc = nn.Sequential(wn_func(Linear(hidden_size, inner_linear, groups=inner_groups)),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                wn_func(Linear(inner_linear, hidden_size, groups=inner_groups)))

    def set_mask(self, mask):
        self.attention.set_mask_q(mask)
        self.attention.set_mask_k(mask)

    def forward(self, inputs):
        x = inputs
        res = x
        x, _ = self.attention(x, x, x)
        x = self.dropout(x).add_(res)
        x = self.lnorm1(x) if hasattr(self, 'lnorm1') else x
        res = x
        x = self.fc(x)
        x = self.dropout(x).add_(res)
        x = self.lnorm2(x) if hasattr(self, 'lnorm2') else x
        return x


class DecoderBlock(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, inner_linear=2048, inner_groups=1,
                 layer_norm=True, weight_norm=False, dropout=0, stateful=None):

        super(DecoderBlock, self).__init__()
        wn_func = wn if weight_norm else lambda x: x
        if layer_norm:
            self.lnorm1 = nn.LayerNorm(hidden_size)
            self.lnorm2 = nn.LayerNorm(hidden_size)
            self.lnorm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.weight_norm = weight_norm
        self.stateful = stateful
        self.attention = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout=dropout, causal=False, weight_norm=weight_norm)
        if stateful is not None:
            residual = False
            if stateful.endswith('_res'):
                stateful = stateful.replace('_res','')
                residual = True
            if stateful in ['RNN', 'iRNN', 'LSTM', 'GRU']:
                self.state_block = Recurrent(stateful, hidden_size, hidden_size,
                                             dropout=dropout, residual=residual, batch_first=True)
            else:
                self.state_block = AverageNetwork(
                    hidden_size, hidden_size, layer_norm=layer_norm, weight_norm=weight_norm, batch_first=True)
        else:
            self.masked_attention = MultiHeadAttention(
                hidden_size, hidden_size, num_heads, dropout=dropout, causal=True, weight_norm=weight_norm)

        self.fc = nn.Sequential(wn_func(Linear(hidden_size, inner_linear, groups=inner_groups)),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                wn_func(Linear(inner_linear, hidden_size, groups=inner_groups)))

    def set_mask(self, mask, context_mask=None):
        if context_mask is not None:
            self.attention.set_mask_k(context_mask)
        if hasattr(self, 'masked_attention'):
            self.masked_attention.set_mask_q(mask)
            self.masked_attention.set_mask_k(mask)

    def forward(self, inputs, context, state=None):
        x = inputs
        res = x
        if self.stateful:
            x, state = self.state_block(x, state)
        else:  # block_state are past inputs
            if state is None:
                x_past = x
            else:
                x_past = torch.cat((state, x), 1)
            x, _ = self.masked_attention(x, x_past, x_past)
            state = x_past
        x = self.dropout(x).add(res)
        x = self.lnorm1(x) if hasattr(self, 'lnorm1') else x
        res = x
        x, attn_enc = self.attention(x, context, context)
        x = self.dropout(x).add_(res)
        x = self.lnorm2(x) if hasattr(self, 'lnorm2') else x
        res = x
        x = self.fc(x)
        x = self.dropout(x).add_(res)
        x = self.lnorm3(x) if hasattr(self, 'lnorm3') else x

        return x, attn_enc, state

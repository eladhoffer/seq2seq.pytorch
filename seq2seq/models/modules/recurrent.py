import torch
import torch.nn as nn
from .attention import AttentionLayer
from .weight_drop import WeightDrop
from .weight_norm import weight_norm as wn
from torch.nn.utils.rnn import PackedSequence


def Recurrent(mode, input_size, hidden_size,
              num_layers=1, bias=True, batch_first=False,
              dropout=0, weight_norm=False, weight_drop=0, bidirectional=False, residual=False,
              zoneout=None, attention_layer=None, forget_bias=None):
    params = dict(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers, bias=bias, batch_first=batch_first,
                  dropout=dropout, bidirectional=bidirectional)
    need_to_wrap = attention_layer is not None \
        or zoneout is not None \
        or mode not in ['LSTM', 'GRU', 'RNN', 'iRNN']
    wn_func = wn if weight_norm else lambda x: x

    if need_to_wrap:
        if mode == 'LSTM':
            rnn_cell = nn.LSTMCell
        elif mode == 'GRU':
            rnn_cell = nn.GRUCell
        else:
            raise Exception('Mode {} is unsupported yet'.format(mode))
        cell = rnn_cell
        if zoneout is not None:
            cell = wrap_zoneout_cell(cell, zoneout)

        if bidirectional:
            bi_module = ConcatRecurrent()
            bi_module.add_module('0', TimeRecurrentCell(wn_func(cell(input_size, hidden_size)),
                                                        batch_first=batch_first,
                                                        lstm=mode == 'LSTM',
                                                        with_attention=attention_layer is not None))
            bi_module.add_module('0.reversed', TimeRecurrentCell(wn_func(cell(input_size, hidden_size)),
                                                                 batch_first=batch_first,
                                                                 lstm=mode == 'LSTM',
                                                                 reverse=True,
                                                                 with_attention=attention_layer is not None))
            module = StackedRecurrent(residual)
            for i in range(num_layers):
                module.add_module(str(i), bi_module)

        else:
            if attention_layer is None:
                cell = StackedCell(rnn_cell=cell,
                                   input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   residual=residual,
                                   weight_norm=weight_norm,
                                   dropout=dropout)
            else:
                cell = StackedsAttentionCell(rnn_cell=cell,
                                             input_size=input_size,
                                             hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             residual=residual,
                                             weight_norm=weight_norm,
                                             dropout=dropout,
                                             attention_layer=attention_layer)
            module = TimeRecurrentCell(cell,
                                       batch_first=batch_first,
                                       lstm=mode == 'LSTM',
                                       with_attention=attention_layer is not None)

    else:
        if mode == 'LSTM':
            rnn = nn.LSTM
        elif mode == 'GRU':
            rnn = nn.GRU
        elif mode == 'RNN':
            rnn = nn.RNN
            params['nonlinearity'] = 'tanh'
        elif mode == 'iRNN':
            rnn = nn.RNN
            params['nonlinearity'] = 'relu'
        else:
            raise Exception('Unknown mode: {}'.format(mode))
        if residual:
            rnn = wrap_stacked_recurrent(rnn,
                                         num_layers=num_layers,
                                         weight_norm=weight_norm,
                                         residual=True)
            params['num_layers'] = 1
        if params.get('num_layers', 0) == 1:
            params.pop('dropout', None)
        module = wn_func(rnn(**params))

    if mode == 'LSTM' and forget_bias is not None:
        for n, p in module.named_parameters():
            if 'bias_hh' in n or 'bias_ih' in n:
                forget_bias_params = p.data.chunk(4)[1]
                forget_bias_params.fill_(forget_bias / 2)
    if mode == 'iRNN':
        for n, p in module.named_parameters():
            if 'weight_hh' in n:
                p.detach().copy_(torch.eye(*p.shape))
    return module


def wrap_stacked_recurrent(recurrent_func, num_layers=1, residual=False, weight_norm=False):
    def f(*kargs, **kwargs):
        module = StackedRecurrent(residual)
        for i in range(num_layers):
            rnn = recurrent_func(*kargs, **kwargs)
            if weight_norm:
                rnn = wn(rnn)
            module.add_module(str(i), rnn)
        return module
    return f


class StackedRecurrent(nn.Sequential):

    def __init__(self, dropout=0, residual=False):
        super(StackedRecurrent, self).__init__()
        self.residual = residual
        self.dropout = dropout

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self))
        next_hidden = []
        for i, module in enumerate(self._modules.values()):
            output, h = module(inputs, hidden[i])
            next_hidden.append(h)
            if self.residual and inputs.size(-1) == output.size(-1):
                inputs = output + inputs
            else:
                inputs = output
            if isinstance(inputs, PackedSequence):
                data = nn.functional.dropout(
                    inputs.data, self.dropout, self.training)
                inputs = PackedSequence(data, inputs.batch_sizes)
            else:
                inputs = nn.functional.dropout(
                    inputs, self.dropout, self.training)

        return output, tuple(next_hidden)


class ConcatRecurrent(nn.Sequential):

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self))
        next_hidden = []
        outputs = []
        for i, module in enumerate(self._modules.values()):
            curr_output, h = module(inputs, hidden[i])
            outputs.append(curr_output)
            next_hidden.append(h)
        output = torch.cat(outputs, -1)
        return output, tuple(next_hidden)


class StackedCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.LSTMCell, residual=False, weight_norm=False):
        super(StackedCell, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.residual = residual
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            rnn = rnn_cell(input_size, hidden_size, bias=bias)
            if weight_norm:
                rnn = wn(rnn_cell)
            self.layers.append(rnn)
            input_size = hidden_size

    def forward(self, inputs, hidden):
        def select_layer(h_state, i):  # To work on both LSTM / GRU, RNN
            if isinstance(h_state, tuple):
                return tuple([select_layer(s, i) for s in h_state])
            else:
                return h_state[i]

        next_hidden = []
        for i, layer in enumerate(self.layers):
            next_hidden_i = layer(inputs, select_layer(hidden, i))
            output = next_hidden_i[0] if isinstance(next_hidden_i, tuple) \
                else next_hidden_i
            if i + 1 < self.num_layers:
                output = self.dropout(output)
            if self.residual and inputs.size(-1) == output.size(-1):
                inputs = output + inputs
            else:
                inputs = output
            next_hidden.append(next_hidden_i)
        if isinstance(hidden, tuple):
            next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
        else:
            next_hidden = torch.stack(next_hidden)
        return inputs, next_hidden


class StackedsAttentionCell(StackedCell):

    def __init__(self, input_size, hidden_size, attention_layer, num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.LSTMCell, residual=False, weight_norm=False):
        super(StackedsAttentionCell, self).__init__(input_size, hidden_size, num_layers,
                                                    dropout, bias, rnn_cell, residual)
        self.attention = attention_layer

    def forward(self, input_with_context, hidden, get_attention=False):
        inputs, context = input_with_context
        if isinstance(context, tuple):
            context_keys, context_values = context
        else:
            context_keys = context_values = context
        hidden_cell, hidden_attention = hidden
        inputs = torch.cat([inputs, hidden_attention], inputs.dim() - 1)
        output_cell, hidden_cell = super(
            StackedsAttentionCell, self).forward(inputs, hidden_cell)
        output, score = self.attention(
            output_cell, context_keys, context_values)
        if get_attention:
            return output, (hidden_cell, output), score
        else:
            del score
            return output, (hidden_cell, output)


def wrap_zoneout_cell(cell_func, zoneout_prob=0):
    def f(*kargs, **kwargs):
        return ZoneOutCell(cell_func(*kargs, **kwargs), zoneout_prob)
    return f


class ZoneOutCell(nn.Module):

    def __init__(self, cell, zoneout_prob=0):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_prob = zoneout_prob

    def forward(self, inputs, hidden):
        def zoneout(h, next_h, prob):
            if isinstance(h, tuple):
                num_h = len(h)
                if not isinstance(prob, tuple):
                    prob = tuple([prob] * num_h)
                return tuple([zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])
            mask = h.new_tensor(h.size()).bernoulli_(prob)
            return mask * next_h + (1 - mask) * h

        next_hidden = self.cell(inputs, hidden)
        next_hidden = zoneout(hidden, next_hidden, self.zoneout_prob)
        return next_hidden


def wrap_time_cell(cell_func, batch_first=False, lstm=True, with_attention=False, reverse=False):
    def f(*kargs, **kwargs):
        return TimeRecurrentCell(cell_func(*kargs, **kwargs), batch_first, lstm, with_attention, reverse)
    return f


class TimeRecurrentCell(nn.Module):

    def __init__(self, cell, batch_first=False, lstm=True, with_attention=False, reverse=False):
        super(TimeRecurrentCell, self).__init__()
        self.cell = cell
        self.lstm = lstm
        self.reverse = reverse
        self.batch_first = batch_first
        self.with_attention = with_attention

    def forward(self, inputs, hidden=None, context=None, mask_attention=None, get_attention=False):
        if self.with_attention and mask_attention is not None:
            self.cell.attention.set_mask(mask_attention)
        hidden_size = self.cell.hidden_size
        batch_dim = 0 if self.batch_first else 1
        time_dim = 1 if self.batch_first else 0
        batch_size = inputs.size(batch_dim)

        if hidden is None:
            num_layers = getattr(self.cell, 'num_layers', 1)
            zero = inputs.data.new(1).zero_()
            h0 = zero.view(1, 1, 1).expand(num_layers, batch_size, hidden_size)
            hidden = h0
            if self.lstm:
                hidden = (hidden, h0)
        if self.with_attention and \
            (not isinstance(hidden, tuple)
             or self.lstm and not isinstance(hidden[0], tuple)):
            # check if additional initial attention state is needed
            zero = inputs.data.new(1).zero_()
            attn_size = self.cell.attention.output_size
            a0 = zero.view(1, 1).expand(batch_size, attn_size)
            hidden = (hidden, a0)

        outputs = []
        attentions = []
        inputs_time = inputs.split(1, time_dim)
        if self.reverse:
            inputs_time.reverse()
        for input_t in inputs_time:
            input_t = input_t.squeeze(time_dim)
            if self.with_attention:
                input_t = (input_t, context)
            if self.with_attention and get_attention:
                output_t, hidden, attn = self.cell(
                    input_t, hidden, get_attention=True)
                attentions += [attn]
            else:
                output_t, hidden = self.cell(input_t, hidden)

            outputs += [output_t]
        if self.reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, time_dim)
        if get_attention:
            attentions = torch.stack(attentions, time_dim)
            return outputs, hidden, attentions
        else:
            return outputs, hidden


class RecurrentAttention(nn.Module):

    def __init__(self, input_size, context_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False, dropout=0,
                 concat_attention=True, num_pre_attention_layers=None,
                 mode='LSTM', residual=False, weight_norm=False, attention=None, forget_bias=None):
        super(RecurrentAttention, self).__init__()

        self.hidden_size = hidden_size
        self.layers = num_layers
        self.concat_attention = concat_attention
        self.batch_first = batch_first
        if isinstance(context_size, tuple):
            context_key_size, context_value_size = context_size
        else:
            context_key_size = context_value_size = context_size
        attention = attention or {}
        attention['key_size'] = context_key_size
        attention['value_size'] = context_value_size
        attention['query_size'] = hidden_size
        attention['batch_first'] = batch_first
        attention['weight_norm'] = weight_norm
        self.attn = AttentionLayer(**attention)
        num_pre_attention_layers = num_pre_attention_layers or num_layers

        if concat_attention and num_pre_attention_layers > 0:
            input_size = input_size + self.attn.output_size
            embedd_attn = self.attn
            del self.attn  # don't include attention layer twice in model
        else:
            embedd_attn = None

        self.rnn_att = Recurrent(mode, input_size, hidden_size,
                                 num_layers=num_pre_attention_layers,
                                 bias=bias, dropout=dropout, forget_bias=forget_bias,
                                 residual=residual, weight_norm=weight_norm, attention_layer=embedd_attn)

        if num_layers > num_pre_attention_layers:
            self.rnn_no_att = Recurrent(mode, hidden_size, hidden_size,
                                        num_layers=num_layers - num_pre_attention_layers, bias=bias,
                                        batch_first=batch_first, residual=residual, weight_norm=weight_norm,
                                        dropout=dropout, forget_bias=forget_bias)

    def forward(self, inputs, context, hidden=None, mask_attention=None, get_attention=False):
        if isinstance(context, tuple):
            context_keys, context_values = context
        else:
            context_keys = context_values = context
        if hasattr(self, 'rnn_no_att'):
            if hidden is None:
                hidden = [None] * 2
            hidden, hidden_2 = hidden

        if not self.concat_attention:
            outputs, hidden = self.rnn_att(inputs, hidden)
            self.attn.set_mask(mask_attention)
            outputs, attentions = self.attn(
                outputs, context_keys, context_values)
        else:
            out = self.rnn_att(inputs, hidden, context,
                               mask_attention=mask_attention,
                               get_attention=get_attention)
            if get_attention:
                outputs, hidden, attentions = out
            else:
                outputs, hidden = out

        if hasattr(self, 'rnn_no_att'):
            outputs, hidden_2 = self.rnn_no_att(outputs, hidden_2)
            hidden = (hidden, hidden_2)

        if get_attention:
            return outputs, hidden, attentions
        else:
            return outputs, hidden

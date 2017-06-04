from torch.autograd import Function, NestedIOFunction, Variable
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
try:
    import torch.backends.cudnn.rnn
except ImportError:
    pass

class LinearPTZGRAD(Function):

    # bias is an optional argument
    def forward(self, input, weight, bias=None):
        self.save_for_backward(input, weight, bias)
        #print(input.size(),weight.size())
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = self.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if self.needs_input_grad[0]:
            mask=grad_output.gt(1e-4).add(grad_output.lt(-1e-4))
            #if mask.sum()>0:
            #    print(mask.float().sum()/mask.numel())
            grad_output.mul_(mask.float())
            grad_input = grad_output.mm(weight)
        if self.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
def linear_ptzgrad(input, weight, bias=None):
    # First braces create a Function object. Any arguments given here
    # will be passed to __init__. Second braces will invoke the __call__
    # operator, that will then use forward() to compute the result and
    # return it.
    return LinearPTZGRAD()(input, weight, bias)

def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
        return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
        return hy


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hx, cx = hidden
        gates = linear_ptzgrad(input, w_ih, b_ih) + linear_ptzgrad(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy


def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert(len(weight) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, 2)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward

def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def AutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False, dropout_state=None):

    if mode == 'RNN_RELU':
        cell = RNNReLUCell
    elif mode == 'RNN_TANH':
        cell = RNNTanhCell
    elif mode == 'LSTM':
        cell = LSTMCell
    elif mode == 'GRU':
        cell = GRUCell
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    if bidirectional:
        layer = (Recurrent(cell), Recurrent(cell, reverse=True))
    else:
        layer = (Recurrent(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


class CudnnRNN(NestedIOFunction):
    def __init__(self, mode, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0,  train=True, bidirectional=False, dropout_state=None):
        super(CudnnRNN, self).__init__()
        if dropout_state is None:
            dropout_state = {}
        self.mode = cudnn.rnn.get_cudnn_mode(mode)
        self.input_mode = cudnn.CUDNN_LINEAR_INPUT
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.train = train
        self.bidirectional = 1 if bidirectional else 0
        self.num_directions = 2 if bidirectional else 1
        self.dropout_seed = torch.IntTensor(1).random_()[0]
        self.dropout_state = dropout_state

    def forward_extended(self, input, weight, hx):

        assert(cudnn.is_acceptable(input))

        output = input.new()

        if torch.is_tensor(hx):
            hy = hx.new()
        else:
            hy = tuple(h.new() for h in hx)

        cudnn.rnn.forward(self, input, hx, weight, output, hy)

        self.save_for_backward(input, hx, weight, output)
        return output, hy


    def backward_extended(self, grad_output, grad_hy):
        input, hx, weight, output = self.saved_tensors

        grad_input, grad_weight, grad_hx = None, None, None

        assert(cudnn.is_acceptable(input))

        grad_input = input.new()
        grad_weight = input.new()
        grad_hx = input.new()
        if torch.is_tensor(hx):
            grad_hx = input.new()
        else:
            grad_hx = tuple(h.new() for h in hx)

        cudnn.rnn.backward_grad(
            self,
            input,
            hx,
            weight,
            output,
            grad_output,
            grad_hy,
            grad_input,
            grad_hx)

        if self.needs_input_grad[1]:
            grad_weight = [tuple(w.new().resize_as_(w).zero_() for w in layer_weight) for layer_weight in weight]
            cudnn.rnn.backward_weight(
                self,
                input,
                hx,
                output,
                weight,
                grad_weight)

        return grad_input, grad_weight, grad_hx


def RNN(*args, **kwargs):
    def forward(input, *fargs, **fkwargs):
        #if cudnn.is_acceptable(input.data):
        #    func = CudnnRNN(*args, **kwargs)
        #else:
        func = AutogradRNN(*args, **kwargs)
        return func(input, *fargs, **fkwargs)

    return forward

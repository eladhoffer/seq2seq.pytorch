import torch
import torch.nn as nn
from .seq2seq import Seq2Seq
from .recurrent import RecurrentAttention


class GNMT(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0, gpu_assignment={'encoder': 1, 'decoder': 2}):
        super(GNMT, self).__init__()
        self.gpu_assignment = gpu_assignment
        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size,
                                                num_layers, bias, batch_first,
                                                dropout)
        if self.gpu_assignment is not None:
            self.encoder.cuda(self.gpu_assignment.get('encoder', 0))
        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size=hidden_size,
                                                num_layers=num_layers,
                                                bias=bias,
                                                batch_first=batch_first,
                                                dropout=dropout)
        if self.gpu_assignment is not None:
            self.decoder.cuda(self.gpu_assignment.get('decoder', 0))

    def bridge(self, context):
        context, _ = context
        return (context, None)

    def forward(self, input_encoder, input_decoder, encoder_state=None):
        if self.gpu_assignment is not None:
            input_encoder = input_encoder.cuda(
                self.gpu_assignment.get('encoder', 0))
            if encoder_state is not None:
                input_encoder = input_encoder.cuda(
                    self.gpu_assignment.get('encoder', 0))

        context = self.encode(input_encoder, encoder_state)
        context = self.bridge(context)

        if self.gpu_assignment is not None:
            input_decoder = input_decoder.cuda(
                self.gpu_assignment.get('decoder', 0))
            context = (context[0].cuda(self.gpu_assignment.get('decoder', 0)),
                       context[1])

        output, _ = self.decode(input_decoder, context)
        if self.gpu_assignment is not None:
            output = output.cuda(self.gpu_assignment.get('output', 0))
        return output


class ResidualRecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0):
        super(ResidualRecurrentEncoder, self).__init__()
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                       bidirectional=True))
        self.rnn_layers.append(nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                       bidirectional=False))
        for n in range(num_layers - 2):
            self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout,
                                           bidirectional=False))
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.vocab_size = vocab_size

    def forward(self, inputs, hidden=None):
        x = self.embedder(inputs)
        x, _ = self.rnn_layers[0](x)
        x, _ = self.rnn_layers[1](x)
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x, hidden = self.rnn_layers[i](x)
            x = x + residual
        return x, hidden


class ResidualRecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128,
                 num_layers=8, bias=True, batch_first=False,
                 dropout=0):
        super(ResidualRecurrentDecoder, self).__init__()

        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(RecurrentAttention(
            hidden_size, hidden_size, num_layers=1))
        self.rnn_layers.append(
            nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout))
        for _ in range(num_layers - 2):
            self.rnn_layers.append(
                nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=bias, dropout=dropout))
        self.embedder = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, inputs, context, get_attention=False):
        enc_context, hidden = context
        hidden = hidden or [None] * len(self.rnn_layers)
        next_hidden = []
        x = self.embedder(inputs)
        if get_attention:
            x, h, attentions = self.rnn_layers[0](x, hidden[0], enc_context,
                                                       get_attention=True)
        else:
            x, h = self.rnn_layers[0](x, hidden[1], enc_context)
        next_hidden.append(h)
        dec_context = x
        x, h = self.rnn_layers[1](x)
        next_hidden.append(h)
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = torch.cat([x, dec_context], 2)
            x, h = self.rnn_layers[i](x, hidden[i])
            next_hidden.append(h)
            x = x + residual
        x = x.view(-1, x.size(2))
        x = self.classifier(x)
        x = x.view(inputs.size(0), inputs.size(1), -1)
        if get_attention:
            return x, (enc_context, tuple(next_hidden)), attentions
        else:
            return x, (enc_context, tuple(next_hidden))


# # #test:
# model = GNMT(32000, gpu_assignment=None)
# x1 = torch.autograd.Variable(torch.rand(16, 32).long().fill_(2))
# x2 = torch.autograd.Variable(torch.rand(7, 32).long().fill_(2))
# y = model(x1, x2)

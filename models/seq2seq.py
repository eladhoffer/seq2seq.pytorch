import torch.nn as nn


class Seq2Seq(nn.Module):

    def __init__(self, encoder=None, decoder=None, bridge=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if bridge is not None:
            self.bridge = bridge

    def encode(self, inputs, hidden=None):
        return self.encoder(inputs, hidden)

    def decode(self, inputs, context):
        return self.decoder(inputs, context)

    def forward(self, input_encoder, input_decoder, encoder_hidden=None):
        context = self.encode(input_encoder, encoder_hidden)
        hidden = None
        if hasattr(self, 'bridge'):
            context = self.bridge(context)
        output, hidden = self.decode(input_decoder, context)
        return output

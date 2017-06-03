import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, bridge=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bridge = bridge

    def encode(self, inputs, state=None):
        if state:
            return self.encoder(inputs, state)
        else:
            return self.encoder(inputs)

    def decode(self, inputs, state=None):
        return self.decoder(inputs, state)

    def forward(self, input_encoder, input_decoder):
        output, state = self.encode(input_encoder)
        if self.bridge is not None:
            input_decoder, state = self.bridge(input_decoder, state)
        return self.decode(input_decoder, state)

import torch.nn as nn
from torch.nn.parallel import data_parallel


class Seq2Seq(nn.Module):

    def __init__(self, encoder=None, decoder=None, bridge=None, batch_first=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if bridge is not None:
            self.bridge = bridge
        self.batch_first = batch_first

    def encode(self, inputs, hidden=None, devices=None):
        if isinstance(devices, tuple):
            return data_parallel(self.encoder, (inputs, hidden),
                                 device_ids=devices,
                                 dim=0 if self.batch_first else 1)
        else:
            return self.encoder(inputs, hidden)

    def decode(self, inputs, context, devices=None):
        if isinstance(devices, tuple):
            return data_parallel(self.decoder, (inputs, context),
                                 device_ids=devices,
                                 dim=0 if self.batch_first else 1)
        else:
            return self.decoder(inputs, context)

    def forward(self, input_encoder, input_decoder, encoder_hidden=None, devices=None):
        if not isinstance(devices, dict):
            devices = {'encoder': devices, 'decoder': devices}
        context = self.encode(input_encoder, encoder_hidden,
                              devices=devices.get('encoder', None))
        if hasattr(self, 'bridge'):
            context = self.bridge(context)
        output, hidden = self.decode(
            input_decoder, context, devices=devices.get('decoder', None))
        return output

    def generate(self, inputs, context):
        return self.decode(inputs, context)

    def clear_state(self):
        pass

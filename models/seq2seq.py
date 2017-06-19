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

    def encode(self, inputs, hidden=None, device_ids=None, output_device=None):
        if device_ids is None:
            return self.encoder(inputs, hidden)
        else:
            return data_parallel(self.encoder, (inputs, hidden),
                                 device_ids=device_ids,
                                 output_device=output_device,
                                 dim=0 if self.batch_first else 1)

    def decode(self, inputs, context, device_ids=None, output_device=None):
        if device_ids is None:
            return self.decoder(inputs, context)
        else:
            return data_parallel(self.decoder, (inputs, context),
                                 device_ids=device_ids,
                                 output_device=output_device,
                                 dim=0 if self.batch_first else 1)

    def forward(self, input_encoder, input_decoder, encoder_hidden=None, device_ids=None, output_device=None):
        context = self.encode(input_encoder, encoder_hidden,
                              device_ids=device_ids, output_device=output_device)
        if hasattr(self, 'bridge'):
            context = self.bridge(context)
        output, hidden = self.decode(
            input_decoder, context, device_ids=device_ids, output_device=output_device)
        return output

    def generate(self, inputs, context):
        return self.decode(inputs, context)

    def clear_state(self):
        pass

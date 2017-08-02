import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from seq2seq.tools.utils import batch_sequences
from .modules.state import State


class Seq2Seq(nn.Module):

    def __init__(self, encoder=None, decoder=None, bridge=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        if bridge is not None:
            self.bridge = bridge

    def bridge(self, context):
        return State(context=context,
                     batch_first=getattr(self.decoder, 'batch_first', context.batch_first))

    def encode(self, inputs, hidden=None, devices=None):
        if isinstance(devices, tuple):
            return data_parallel(self.encoder, (inputs, hidden),
                                 device_ids=devices,
                                 dim=0 if self.encoder.batch_first else 1)
        else:
            return self.encoder(inputs, hidden)

    def decode(self, inputs, state, get_attention=None, devices=None):
        if isinstance(devices, tuple):
            inputs = (inputs, state, get_attention) if get_attention else (
                inputs, state)
            return data_parallel(self.decoder, inputs,
                                 device_ids=devices,
                                 dim=0 if self.decoder.batch_first else 1)
        else:
            if get_attention:
                return self.decoder(inputs, state, get_attention=get_attention)
            else:
                return self.decoder(inputs, state)

    def forward(self, input_encoder, input_decoder, encoder_hidden=None, devices=None):
        if not isinstance(devices, dict):
            devices = {'encoder': devices, 'decoder': devices}
        context = self.encode(input_encoder, encoder_hidden,
                              devices=devices.get('encoder', None))
        if hasattr(self, 'bridge'):
            state = self.bridge(context)
        output, state = self.decode(
            input_decoder, state, devices=devices.get('decoder', None))
        return output

    def generate(self, input_list, state_list, k=1, feed_all_timesteps=False, get_attention=False):
        # assert isinstance(input_list, list) or isinstance(input_list, tuple)
        # assert isinstance(input_list[0], list) or isinstance(
            # input_list[0], tuple)

        view_shape = (-1, 1) if self.decoder.batch_first else (1, -1)
        time_dim = 1 if self.decoder.batch_first else 0

        # For recurrent models, the last input frame is all we care about,
        # use feed_all_timesteps whenever the whole input needs to be fed
        if feed_all_timesteps:
            inputs = [torch.LongTensor(inp) for inp in input_list]
            inputs = batch_sequences(
                inputs, batch_first=self.decoder.batch_first)[0]
        else:
            inputs = torch.LongTensor(
                [inputs[-1] for inputs in input_list]).view(*view_shape)

        inputs_var = Variable(inputs, volatile=True)
        if next(self.decoder.parameters()).is_cuda:
            inputs_var = inputs_var.cuda()

        states = State().from_list(state_list)
        logits, new_states = self.decode(
            inputs_var, states, get_attention=get_attention)
        # use only last prediction
        logits = logits.select(time_dim, -1).contiguous()
        logprobs = log_softmax(logits.view(-1, logits.size(-1)))
        logprobs, words = logprobs.data.topk(k, 1)
        new_states_list = [new_states[i] for i in range(len(input_list))]
        return words, logprobs, new_states_list

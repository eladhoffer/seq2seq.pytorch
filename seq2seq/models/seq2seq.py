import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from seq2seq.tools.utils import batch_padded_sequences


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

    def decode(self, inputs, context, get_attention=None, devices=None):
        if isinstance(devices, tuple):
            inputs = (inputs, context, get_attention) if get_attention else (
                inputs, context)
            return data_parallel(self.decoder, inputs,
                                 device_ids=devices,
                                 dim=0 if self.batch_first else 1)
        else:
            if get_attention:
                return self.decoder(inputs, context, get_attention=get_attention)
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

    def clear_state(self):
        pass

    def generate(self, input_list, state_list, k=1, feed_all_timesteps=False, get_attention=False):
        # assert isinstance(input_list, list) or isinstance(input_list, tuple)
        # assert isinstance(input_list[0], list) or isinstance(
            # input_list[0], tuple)

        time_dim = 1 if self.batch_first else 0
        view_shape = (-1, 1) if self.batch_first else (1, -1)

        # For recurrent models, the last input frame is all we care about,
        # use feed_all_timesteps whenever the whole input needs to be fed
        if feed_all_timesteps:
            inputs = [torch.LongTensor(inp) for inp in input_list]
            inputs = batch_padded_sequences(
                inputs, batch_first=self.batch_first)
        else:
            inputs = torch.LongTensor(
                [inputs[-1] for inputs in input_list]).view(*view_shape)

        inputs_var = Variable(inputs, volatile=True)
        if next(self.decoder.parameters()).is_cuda:
            inputs_var = inputs_var.cuda()
        states = self.merge_states(state_list)

        if get_attention:
            logits, new_states, attention = self.decode(
                inputs_var, states, get_attention=True)
            attention = attention.select(time_dim, -1).data
        else:
            attention = None
            logits, new_states = self.decode(inputs_var, states)
        # use only last prediction
        logits = logits.select(time_dim, -1).contiguous()
        logprobs = log_softmax(logits.view(-1, logits.size(-1)))
        logprobs, words = logprobs.data.topk(k, 1)
        new_states = [self.select_state(new_states, i)
                      for i in range(len(input_list))]
        return words, logprobs, new_states, attention

    def merge_states(self, state_list):
        if isinstance(state_list[0], tuple):
            return tuple([self.merge_states(s) for s in zip(*state_list)])
        else:
            if state_list[0] is None:
                return None
            if state_list[0].dim() == 3 and not self.batch_first:
                batch_dim = 1
            else:
                batch_dim = 0
            return torch.cat(state_list, batch_dim)

    def select_state(self, state, i):
        if isinstance(state, tuple):
            return tuple(self.select_state(s, i) for s in state)
        else:
            if state is None:
                return None
            if state.dim() == 3 and not self.batch_first:
                batch_dim = 1
            else:
                batch_dim = 0
            return state.narrow(batch_dim, i, 1)

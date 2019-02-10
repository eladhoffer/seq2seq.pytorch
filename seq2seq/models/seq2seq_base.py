import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torch.nn.functional import log_softmax
from seq2seq.tools import batch_sequences
from .modules.state import State
from seq2seq.tools.config import UNK, PAD
from seq2seq.tools.beam_search import SequenceGenerator


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

    def encode(self, inputs, hidden=None, device_ids=None):
        if isinstance(device_ids, tuple):
            return data_parallel(self.encoder, (inputs, hidden),
                                 device_ids=device_ids,
                                 dim=0 if self.encoder.batch_first else 1)
        else:
            return self.encoder(inputs, hidden)

    def decode(self, inputs, state, get_attention=None, device_ids=None):
        if isinstance(device_ids, tuple):
            inputs = (inputs, state, get_attention) if get_attention else (
                inputs, state)
            return data_parallel(self.decoder, inputs,
                                 device_ids=device_ids,
                                 dim=0 if self.decoder.batch_first else 1)
        else:
            if get_attention:
                return self.decoder(inputs, state, get_attention=get_attention)
            else:
                return self.decoder(inputs, state)

    def forward(self, input_encoder, input_decoder, encoder_hidden=None, device_ids=None):
        if not isinstance(device_ids, dict):
            device_ids = {'encoder': device_ids, 'decoder': device_ids}
        context = self.encode(input_encoder, encoder_hidden,
                              device_ids=device_ids.get('encoder', None))
        if hasattr(self, 'bridge'):
            state = self.bridge(context)
        output, state = self.decode(
            input_decoder, state, device_ids=device_ids.get('decoder', None))
        return output

    def _decode_step(self, input_list, state_list, k=1,
                     feed_all_timesteps=False,
                     remove_unknown=False,
                     get_attention=False,
                     device_ids=None):

        view_shape = (-1, 1) if self.decoder.batch_first else (1, -1)
        time_dim = 1 if self.decoder.batch_first else 0
        device = next(self.decoder.parameters()).device

        # For recurrent models, the last input frame is all we care about,
        # use feed_all_timesteps whenever the whole input needs to be fed
        if feed_all_timesteps:
            inputs = [torch.tensor(inp, device=device, dtype=torch.long)
                      for inp in input_list]
            inputs = batch_sequences(
                inputs, device=device, batch_first=self.decoder.batch_first)[0]

        else:
            last_tokens = [inputs[-1] for inputs in input_list]
            inputs = torch.stack(last_tokens).view(*view_shape)

        states = State().from_list(state_list)
        logits, new_states = self.decode(inputs, states,
                                         get_attention=get_attention,
                                         device_ids=device_ids)
        # use only last prediction
        logits = logits.select(time_dim, -1).contiguous()
        if remove_unknown:
            # Remove possibility of unknown
            logits[:, UNK].fill_(-float('inf'))
        logprobs = log_softmax(logits, dim=1)
        logprobs, words = logprobs.topk(k, 1)
        new_states_list = [new_states[i] for i in range(len(input_list))]
        return words, logprobs, new_states_list

    def generate(self, input_encoder, input_decoder, beam_size=None,
                 max_sequence_length=None, length_normalization_factor=0,
                 get_attention=False, device_ids=None):
        if not isinstance(device_ids, dict):
            device_ids = {'encoder': device_ids, 'decoder': device_ids}
        context = self.encode(input_encoder,
                              device_ids=device_ids.get('encoder', None))
        if hasattr(self, 'bridge'):
            state = self.bridge(context)
        state_list = state.as_list()
        generator = SequenceGenerator(
            decode_step=self._decode_step,
            beam_size=beam_size,
            max_sequence_length=max_sequence_length,
            get_attention=get_attention,
            length_normalization_factor=length_normalization_factor,
            device_ids=device_ids.get('encoder', None))
        return generator.beam_search(input_decoder, state_list)

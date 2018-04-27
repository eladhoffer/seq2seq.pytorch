import torch
import torch.nn as nn
from .seq2seq_base import Seq2Seq
from .recurrent import RecurrentEncoder, RecurrentAttentionDecoder
from .transformer import TransformerAttentionEncoder, TransformerAttentionDecoder
from .modules.state import State
from seq2seq.tools.config import PAD


class HybridSeq2Seq(Seq2Seq):

    def __init__(self, vocab_size, tie_embedding=False,
                 transfer_hidden=False, encoder=None, decoder=None):
        super(HybridSeq2Seq, self).__init__()
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder.setdefault('vocab_size', vocab_size)

        encoder_type = encoder.pop('type', 'recurrent')
        decoder_type = decoder.pop('type', 'recurrent')

        if encoder_type == 'recurrent':
            self.encoder = RecurrentEncoder(**encoder)
        elif encoder_type == 'transformer':
            self.encoder = TransformerAttentionEncoder(**encoder)
        decoder['context_size'] = self.encoder.hidden_size

        if decoder_type == 'recurrent':
            self.decoder = RecurrentAttentionDecoder(**decoder)

        elif decoder_type == 'transformer':
            self.decoder = TransformerAttentionDecoder(**encoder)
        self.transfer_hidden = transfer_hidden

        if tie_embedding:
            self.encoder.embedder.weight = self.decoder.embedder.weight

    def generate(self, input_list, state_list, k=1, feed_all_timesteps=True, get_attention=False):
        # TODO cache computation, not inputs
        if isinstance(self.decoder, TransformerAttentionDecoder):
            feed_all_timesteps = True
        else:
            feed_all_timesteps = False
        return super(HybridSeq2Seq, self).generate(input_list, state_list, k=k,
                                                   feed_all_timesteps=feed_all_timesteps,
                                                   get_attention=get_attention)

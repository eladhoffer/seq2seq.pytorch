import torch
from random import randrange
from math import floor
from torch.nn.utils.rnn import pack_padded_sequence
from .config import PAD


def _limit_lengths(seqs, max_length=None, max_tokens=None):
    max_length = max_length or float('inf')
    lengths = [min(s.nelement(), max_length) for s in seqs]
    if max_tokens is not None:
        num_tokens = sum(lengths)
        if num_tokens > max_tokens:
            max_length = int(floor(num_tokens / len(seqs)))
            lengths = [min(length, max_length) for length in lengths]
    return lengths


def batch_sequences(seqs, max_length=None, max_tokens=None,
                    batch_first=False, sort=False, pack=False, augment=False):
    """
    seqs: a list of Tensors to be batched together
    max_length: maximum sequence length permitted
    max_tokens: maximum number of tokens in batch permitted

    """
    batch_dim, time_dim = (0, 1) if batch_first else (1, 0)
    if len(seqs) == 1:
        lengths = _limit_lengths(seqs, max_length, max_tokens)
        seq_tensor = seqs[0][:lengths[0]]
        seq_tensor = seq_tensor.unsqueeze(batch_dim)
    else:
        if sort:
            seqs.sort(key=len, reverse=True)
        lengths = _limit_lengths(seqs, max_length, max_tokens)
        batch_length = max(lengths)
        tensor_size = (len(seqs), batch_length) if batch_first \
            else (batch_length, len(seqs))
        seq_tensor = torch.LongTensor(*tensor_size).fill_(PAD)
        for i, seq in enumerate(seqs):
            start_seq = 0
            end_seq = lengths[i]
            if augment and end_seq < seq.nelement():
                delta = randrange(seq.nelement() - end_seq + 1)
                start_seq += delta
                end_seq += delta
            seq_tensor.narrow(time_dim, 0, lengths[i]).select(batch_dim, i)\
                .copy_(seq[start_seq:end_seq])
    if pack:
        seq_tensor = pack_padded_sequence(
            seq_tensor, lengths, batch_first=batch_first)
    return (seq_tensor, lengths)

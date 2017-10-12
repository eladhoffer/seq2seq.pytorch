import torch
from torch.nn.utils.rnn import pack_padded_sequence
from .config import PAD

def batch_sequences(seqs, max_length=None, batch_first=False, sort=False, pack=False):
    max_length = max_length or float('inf')
    batch_dim, time_dim = (0, 1) if batch_first else (1, 0)
    if len(seqs) == 1:
        lengths = [min(len(seqs[0]), max_length)]
        seq_tensor = seqs[0][:lengths[0]]
        seq_tensor = seq_tensor.unsqueeze(batch_dim)
    else:
        if sort:
            seqs.sort(key=len, reverse=True)
        lengths = [min(len(s), max_length) for s in seqs]
        batch_length = max(lengths)
        tensor_size = (len(seqs), batch_length) if batch_first \
            else (batch_length, len(seqs))
        seq_tensor = torch.LongTensor(*tensor_size).fill_(PAD)
        for i, seq in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor.narrow(time_dim, 0, end_seq).select(batch_dim, i)\
                .copy_(seq[:end_seq])
    if pack:
        seq_tensor = pack_padded_sequence(
            seq_tensor, lengths, batch_first=batch_first)
    return (seq_tensor, lengths)

import torch
from random import randrange
from math import floor
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
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


# def _limit_batch_tokens(seqs, max_length=None, max_tokens=None, log=False):
#     """
#     seqs: a list of Tensors to be batched together
#     max_length: maximum sequence length permitted
#     max_tokens: maximum number of tokens (with padding) permitted -- batch will be trimed if exceeded 
#     """
#     max_length = max_length or float('inf')
#     lengths = [min(s.nelement(), max_length) for s in seqs]
#     if max_tokens is not None:
#         num_tokens = max(lengths) * len(seqs)
#         if num_tokens > max_tokens:  # needs to restrict batch size to fit maximum tokens
#             # account for padding in final tensor
#             padded_lengths = np.maximum.accumulate(lengths)
#             num_tokens_batch = padded_lengths * (np.arange(len(seqs)) + 1)
#             # determine new batch size and trim sequence
#             B = int((num_tokens_batch > max_tokens).argmax() - 1)
#             seqs = seqs[:B]
#             lengths = lengths[:B]
#             if log:
#                 logging.debug('Trimmed batch to %s as number of tokens was > %s'
#                               % (B, max_tokens))
#     return seqs, lengths


def batch_sequences(seqs, max_length=None, max_tokens=None, fixed_length=None, batch_first=False, pad_value=PAD,
                    sort=False, pack=False, augment=False, device=None, dtype=torch.long):
    """
    seqs: a list of Tensors to be batched together
    max_length: maximum sequence length permitted
    max_tokens: maximum number of tokens in batch permitted

    """
    batch_dim, time_dim = (0, 1) if batch_first else (1, 0)
    if fixed_length is not None:
        fixed_length = max_length = min(max_length, fixed_length)
    if len(seqs) == 1 and not fixed_length:
        lengths = _limit_lengths(seqs, max_length, max_tokens)
        seq_tensor = seqs[0].view(-1,)[:lengths[0]]
        seq_tensor = seq_tensor.unsqueeze(batch_dim)\
            .to(dtype=dtype, device=device)
    else:
        if sort:
            seqs.sort(key=len, reverse=True)
        lengths = _limit_lengths(seqs, max_length, max_tokens)
        batch_length = max(lengths) if fixed_length is None\
            else fixed_length
        tensor_size = (len(seqs), batch_length) if batch_first \
            else (batch_length, len(seqs))
        seq_tensor = torch.full(tensor_size, pad_value,
                                dtype=dtype, device=device)
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
        if device is not None:  # batch_sizes is not casted to device by default
            seq_tensor = PackedSequence(seq_tensor.data,
                                        seq_tensor.batch_sizes.to(device))
    return (seq_tensor, lengths)


def batch_nested_sequences(seqs_subseqs, max_length=None, max_tokens=None, fixed_length=None, batch_first=True, pad_value=PAD,
                          augment=False, device=None, dtype=torch.long):
    """
    seqs: a list of Tensors to be batched together
    sub_seqs: a list of list of Tensors to be batched together
    max_length: maximum sequence length permitted
    max_tokens: maximum number of tokens in batch permitted

    """
    seqs, sub_seqs = zip(*seqs_subseqs)
    batch_dim, time_dim = (0, 1) if batch_first else (1, 0)
    if fixed_length is not None:
        fixed_length = max_length = min(max_length, fixed_length)
    lengths = _limit_lengths(seqs, max_length, max_tokens)

    sub_seqs = [s[:length] for s, length in zip(sub_seqs, lengths)]
    sub_lengths = [[sub.nelement() for sub in s] for s in sub_seqs]
    batch_length = max(lengths) if fixed_length is None\
        else fixed_length
    batch_sub_length = max([max([s2.numel() for s2 in s1]) for s1 in sub_seqs])
    sub_tensor_size = (len(seqs), batch_length, batch_sub_length) if batch_first \
        else (batch_length, batch_sub_length, len(seqs))
    sub_seq_tensor = torch.full(sub_tensor_size, pad_value,
                                dtype=dtype, device=device)
    tensor_size = (len(seqs), batch_length) if batch_first \
        else (batch_length, len(seqs))
    seq_tensor = torch.full(tensor_size, pad_value,
                            dtype=dtype, device=device)
    for i, seq in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor.narrow(time_dim, 0, lengths[i]).select(batch_dim, i)\
            .copy_(seq[0:end_seq])
        for j, sub_seq in enumerate(sub_seqs[i]):
            end_sub_seq = sub_lengths[i][j]
            sub_seq_tensor\
                .narrow(time_dim+1, 0, end_sub_seq)\
                .select(time_dim, j)\
                .select(batch_dim, i)\
                .copy_(sub_seq[0:end_sub_seq])

    return (seq_tensor, lengths), (sub_seq_tensor, sub_lengths)

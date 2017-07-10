import torch
import torch.nn as nn
import torch.nn.functional as F
""" Implementations of attention layers."""


class AttentionLayer(nn.Module):
    """
    Attention layer according to https://arxiv.org/abs/1409.0473.

    Params:
      num_units: Number of units used in the attention layer
    """

    def __init__(self, query_size, key_size=None, value_size=None, mode='bahdanau',
                 normalize=False, dropout=0, batch_first=False, output_transform=True):
        super(AttentionLayer, self).__init__()
        assert mode == 'bahdanau' or mode == 'dot_prod'
        key_size = key_size or query_size  # Usually key and query sizes are the same
        value_size = value_size or key_size  # Usually key and values are the same
        self.mode = mode
        self.normalize = normalize
        if mode == 'bahdanau':
            self.linear_att = nn.Linear(key_size, 1)
            if normalize:
                self.linear_att = nn.utils.weight_norm(self.linear_att)
        if output_transform:
            self.linear_out = nn.Linear(query_size + key_size, key_size)
        self.linear_q = nn.Linear(query_size, key_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.mask = None

    def set_mask(self, mask):
        # applies a mask of b x t length
        self.mask = mask
        if not self.batch_first:
            self.mask = self.mask.t()

    def calc_score(self, att_query, att_keys):
        """
        att_query is: b x n
        att_keys is b x t x n
        """
        b, t, n = list(att_keys.size())
        if self.mode == 'bahdanau':
            att_query = att_query.unsqueeze(1).expand(b, t, n)
            sum_qk = att_query + att_keys
            sum_qk = sum_qk.view(b * t, n)
            return self.linear_att(F.tanh(sum_qk)).view(b, t)
        elif self.mode == 'dot_prod':
            att_query = att_query.view(b, n, 1)
            out = torch.bmm(att_keys, att_query).view(b, t)
            if self.normalize:
                out = out / (n ** 0.5)
            return out

    def forward(self, query, keys, values=None):
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if values is not None:
                values = values.transpose(0, 1)
        values = values or keys

        b, t, n = list(values.size())

        # Fully connected layers to transform query
        att_query = self.linear_q(query)

        scores = self.calc_score(att_query, keys)  # size b x t
        if self.mask is not None:
            scores.data.masked_fill_(self.mask, -1e12)

        # Normalize the scores
        scores_normalized = F.softmax(scores).unsqueeze(1)  # b x 1 x t

        # Calculate the weighted average of the attention inputs
        # according to the scores
        scores_normalized = self.dropout(scores_normalized)
        context = torch.bmm(scores_normalized, values).squeeze(1)  # b x n

        if hasattr(self, 'linear_out'):
            context = F.tanh(self.linear_out(torch.cat([query, context], 1)))
        return context, scores_normalized


class SDPAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, dropout=0, causal=False):
        super(SDPAttention, self).__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.mask_q = None
        self.mask_k = None

    def set_mask_q(self, masked_tq):
        # applies a mask of b x tq length
        self.mask_q = masked_tq

    def set_mask_k(self, masked_tk):
        # applies a mask of b x tk length
        self.mask_k = masked_tk

    def forward(self, q, k, v):
        b_q, t_q, dim_q = list(q.size())
        b_k, t_k, dim_k = list(k.size())
        b_v, t_v, dim_v = list(v.size())
        assert(b_q == b_k and b_k == b_v)  # batch size should be equal
        assert(dim_q == dim_k)  # dims should be equal
        assert(t_k == t_v)  # times should be equal
        b = b_q
        qk = torch.bmm(q, k.transpose(1, 2))  # b x t_q x t_k
        qk = qk / (dim_k ** 0.5)
        mask = None
        if self.causal:
            causal_mask = q.data.new(t_q, t_k).byte().fill_(1).triu_(1)
            mask = causal_mask.unsqueeze(0).expand(b, t_q, t_k)
        if self.mask_k is not None:
            mask_k = self.mask_k.unsqueeze(1).expand(b, t_q, t_k)
            mask = mask_k if mask is None else mask | mask_k
        if self.mask_q is not None:
            mask_q = self.mask_q.unsqueeze(2).expand(b, t_q, t_k)
            mask = mask_q if mask is None else mask | mask_q
        if mask is not None:
            qk.data.masked_fill_(mask, -1e9)

        sm_qk = F.softmax(qk.view(-1, t_k)).view(b, t_q, t_k)
        sm_qk = self.dropout(sm_qk)
        return torch.bmm(sm_qk, v)  # b x t_q x dim_v


class MultiHeadAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, input_size, output_size, num_heads, dropout=0, causal=False):
        super(MultiHeadAttention, self).__init__()
        assert(input_size % num_heads == 0)
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.linear_q = nn.Linear(input_size, input_size)
        self.linear_k = nn.Linear(input_size, input_size)
        self.linear_v = nn.Linear(input_size, input_size)
        self.linear_out = nn.Linear(input_size, output_size)
        self.sdp_attention = SDPAttention(dropout=dropout, causal=causal)

    def set_mask_q(self, masked_tq):
        # applies a mask of b x tq length
        self.sdp_attention.set_mask_q(masked_tq)

    def set_mask_k(self, masked_tk):
        # applies a mask of b x tk length
        self.sdp_attention.set_mask_k(masked_tk)

    def forward(self, q, k, v):

        b_q, t_q, dim_q = list(q.size())
        b_k, t_k, dim_k = list(k.size())
        b_v, t_v, dim_v = list(v.size())
        qw = self.linear_q(q.view(-1, dim_q)).view(b_q, t_q, dim_q)
        kw = self.linear_k(k.view(-1, dim_k)).view(b_k, t_k, dim_k)
        vw = self.linear_v(v.view(-1, dim_v)).view(b_v, t_v, dim_v)
        qw = qw.chunk(self.num_heads, 2)
        kw = kw.chunk(self.num_heads, 2)
        vw = vw.chunk(self.num_heads, 2)
        output = []
        for i in range(self.num_heads):
            out_h = self.sdp_attention(qw[i], kw[i], vw[i])
            output.append(out_h)
        output = torch.cat(output, 2)

        return self.linear_out(output.view(-1, output.size(2))).view(b_q, t_q, self.output_size)

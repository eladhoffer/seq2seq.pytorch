

import torch
import torch.nn as nn


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.


            H_1 H_2 H_3 ... H_n
              q   q   q       q
                |  |   |       |
                  \ |   |      /
                          .....
                      \   |  /
                              a

    Constructs a unit mapping.
        $$(H_1 + H_n, q) => (a)$$
        Where H is of `batch x n x dim` and q is of `batch x dim`.

        The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

    """

    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn


class SDPAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, causal=False):
        super(SDPAttention, self).__init__()
        self.causal = causal
        self.softmax = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

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
        if self.causal:
            for t2 in range(t_k):
                for t1 in range(t_q):
                    if t2 > t1:
                        qk[:, t1, t2].data.fill_(-float('inf'))
        sm_qk = self.softmax(qk.view(-1, t_k)).view(b, t_q, t_k)
        return torch.bmm(sm_qk, v)  # b x t_q x dim_v


class MultiHeadAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, input_size, output_size, num_heads, causal=False):
        super(MultiHeadAttention, self).__init__()
        assert(input_size % num_heads == 0)
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.linear_q = nn.Linear(input_size, input_size)
        self.linear_k = nn.Linear(input_size, input_size)
        self.linear_v = nn.Linear(input_size, input_size)
        self.linear_out = nn.Linear(input_size, output_size)
        self.sdp_attention = SDPAttention(causal=causal)
        self.softmax = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

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

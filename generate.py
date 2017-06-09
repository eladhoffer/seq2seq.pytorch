#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from tools.config import EOS, BOS
from tools.beam_search import SequenceGenerator
import math

cuda = False
checkpoint = torch.load('./results/en_he_onmt/checkpoint.pth.tar')
model = checkpoint['model']
src_tok, target_tok = checkpoint['tokenizers'].values()
bos = Variable(torch.LongTensor([BOS]).view(-1, 1))
if cuda:
    model.cuda()
    bos = bos.cuda()
else:
    model.cpu()


# src = 'hello world, you are crazy'

while True:
    src = input()
    src = src_tok.tokenize(src, insert_start=[BOS], insert_end=[EOS])
    src = Variable(src, volatile=True)

    if cuda:
        src = src.cuda()
    src, state = model.encode(src.view(-1, 1))
    enc_hidden, context, init_output = state

    def decode(x, s):
        out, (s, _, _) = model.decode(x, (s, context, init_output))
        return out, s

    generator = SequenceGenerator(model=model.decode, beam_size=5)

    pred, logporob = generator.beam_search(bos, state)
    sentences = [torch.LongTensor(s[:-1]) for s in pred]
    print(target_tok.detokenize(sentences[0])[::-1])
    # for s,p in zip(sentences,logporob):
    #     print(target_tok.detokenize(s)[::-1],' p=%s' % math.exp(p))

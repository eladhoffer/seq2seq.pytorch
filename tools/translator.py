#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from tools.config import EOS, BOS, LANGUAGE_TOKENS
from tools.beam_search import SequenceGenerator
import math
from tools.quantize import quantize_model, dequantize_model


class Translator(object):

    def __init__(self, model, src_tok, target_tok,
                 insert_src_start=[BOS],
                 insert_src_end=[EOS],
                 insert_target_start=[BOS],
                 beam_size=5,
                 length_normalization_factor=0,
                 cuda=False):
        self.model = model
        self.src_tok = src_tok
        self.target_tok = target_tok
        self.insert_target_start = insert_target_start
        self.insert_src_start = insert_src_start
        self.insert_src_end = insert_src_end
        self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        model.eval()
        self.generator = SequenceGenerator(
            model=self.model.decode,
            beam_size=beam_size,
            length_normalization_factor=length_normalization_factor)

    def translate(self, input_sentence, target_priming=''):
        src = self.src_tok.tokenize(input_sentence,
                                    insert_start=self.insert_src_start,
                                    insert_end=self.insert_src_end)
        bos = self.target_tok.tokenize(
            target_priming, insert_start=self.insert_target_start)
        src = Variable(src, volatile=True)
        bos = Variable(torch.LongTensor([BOS]).view(-1, 1), volatile=True)
        if self.cuda:
            src = src.cuda()
            bos = bos.cuda()
        context = self.model.encode(src.view(-1, 1))
        if hasattr(self.model, 'bridge'):
            context = self.model.bridge(context)
        pred, logporob = self.generator.beam_search(bos, context)
        sentences = [torch.LongTensor(s[:-1]) for s in pred]
        return self.target_tok.detokenize(sentences[0])
        # for s,p in zip(sentences,logporob):
        #     print(target_tok.detokenize(s)[::-1],' p=%s' % math.exp(p))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from tools.config import EOS, BOS, LANGUAGE_TOKENS
from tools.beam_search import SequenceGenerator


class Translator(object):

    def __init__(self, model, src_tok, target_tok,
                 beam_size=5,
                 length_normalization_factor=0,
                 max_sequence_length=50,
                 batch_first=False,
                 cuda=False):
        self.model = model
        self.src_tok = src_tok
        self.target_tok = target_tok
        self.insert_target_start = [BOS]
        self.insert_src_start = [BOS]
        self.insert_src_end = [EOS]
        self.batch_first = batch_first
        self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        model.eval()
        self.generator = SequenceGenerator(
            model=self.model.generate,
            beam_size=beam_size,
            batch_first=batch_first,
            max_sequence_length=max_sequence_length,
            length_normalization_factor=length_normalization_factor)

    def set_src_language(self, language):
        lang = self.src_tok.special_tokens.index(LANGUAGE_TOKENS[language])
        self.insert_src_start = [BOS, lang]

    def set_target_language(self, language):
        lang = self.target_tok.special_tokens.index(LANGUAGE_TOKENS[language])
        self.insert_target_start = [BOS, lang]

    def translate(self, input_sentence, target_priming=''):
        src = self.src_tok.tokenize(input_sentence,
                                    insert_start=self.insert_src_start,
                                    insert_end=self.insert_src_end)
        bos = self.target_tok.tokenize(
            target_priming, insert_start=self.insert_target_start)
        shape = (1, -1) if self.batch_first else (-1, 1)
        src = Variable(src.view(*shape), volatile=True)
        bos = Variable(bos.view(*shape), volatile=True)
        if self.cuda:
            src = src.cuda()
            bos = bos.cuda()
        self.model.clear_state()
        context = self.model.encode(src)
        if hasattr(self.model, 'bridge'):
            context = self.model.bridge(context)
        pred, logporob = self.generator.beam_search(bos, context)
        sentences = [torch.LongTensor(s[:-1]) for s in pred]
        output = self.target_tok.detokenize(sentences[0])
        if len(target_priming) > 0:
            output = ' '.join([target_priming, output])
        return output
        # for s,p in zip(sentences,logporob):
        #     print(target_tok.detokenize(s)[::-1],' p=%s' % math.exp(p))

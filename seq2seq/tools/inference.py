#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from .config import EOS, BOS, LANGUAGE_TOKENS
from .beam_search import SequenceGenerator
from torch.nn.functional import adaptive_avg_pool2d
from .utils import batch_padded_sequences
from .quantize import dequantize_model


class Translator(object):

    def __init__(self, model, src_tok, target_tok,
                 beam_size=5,
                 length_normalization_factor=0,
                 max_sequence_length=50,
                 get_attention=False,
                 cuda=False):
        self.model = model
        self.src_tok = src_tok
        self.target_tok = target_tok
        self.insert_target_start = [BOS]
        self.insert_src_start = [BOS]
        self.insert_src_end = [EOS]
        self.batch_first = getattr(model, 'batch_first', False)
        self.get_attention = get_attention
        self.cuda = cuda
        if getattr(model, 'quantized', False):
            dequantize_model(model)
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        model.eval()
        self.generator = SequenceGenerator(
            model=self.model,
            beam_size=beam_size,
            max_sequence_length=max_sequence_length,
            get_attention=get_attention,
            length_normalization_factor=length_normalization_factor)

    def set_src_language(self, language):
        lang = self.src_tok.special_tokens.index(LANGUAGE_TOKENS(language))
        self.insert_src_start = [BOS, lang]

    def set_target_language(self, language):
        lang = self.target_tok.special_tokens.index(LANGUAGE_TOKENS(language))
        self.insert_target_start = [BOS, lang]

    def translate(self, input_sentences, target_priming=None):
        """input_sentences is either a string or list of strings"""
        if isinstance(input_sentences, list):
            flatten = False
        else:
            input_sentences = [input_sentences]
            flatten = True
        batch = len(input_sentences)
        src_tok = [self.src_tok.tokenize(sentence,
                                         insert_start=self.insert_src_start,
                                         insert_end=self.insert_src_end)
                   for sentence in input_sentences]
        if target_priming is None:
            bos = [self.insert_target_start] * batch
        else:
            if isinstance(target_priming, list):
                bos = [list(self.target_tok.tokenize(priming,
                                                     insert_start=self.insert_target_start))
                       for priming in target_priming]
            else:
                bos = self.target_tok.tokenize(target_priming,
                                               insert_start=self.insert_target_start)
                bos = [list(bos)] * batch

        src = Variable(batch_padded_sequences(
            src_tok, batch_first=self.batch_first), volatile=True)
        if self.cuda:
            src = src.cuda()

        self.model.clear_state()
        context = self.model.encode(src)
        if hasattr(self.model, 'bridge'):
            context = self.model.bridge(context)
        context_list = [self.model.select_state(
            context, i) for i in range(batch)]

        preds, logprobs, attentions = self.generator.beam_search(
            bos, context_list)
        # remove forced  tokens
        preds = [p[len(self.insert_target_start):] for p in preds]
        output = [self.target_tok.detokenize(p[:-1]) for p in preds]

        output = output[0] if flatten else output
        logprobs = logprobs[0] if flatten else logprobs
        if self.get_attention:
            attentions = [torch.stack(att, 1) for att in attentions]
            if target_priming is not None:
                preds = [
                    preds[b][-attentions[b].size(1):] for b in range(batch)]
            attentions = attentions[0] if flatten else attentions

            preds = [[self.target_tok.idx2word(
                idx) for idx in p] for p in preds]
            preds = preds[0] if flatten else preds
            src = [[self.src_tok.idx2word(idx)
                    for idx in list(s)] for s in src_tok]
            src = src[0] if flatten else src
            return output, (attentions, src, preds)
        else:
            return output


class CaptionGenerator(Translator):

    def __init__(self, model, img_transform, target_tok,
                 beam_size=5,
                 length_normalization_factor=0,
                 max_sequence_length=50,
                 get_attention=False,
                 cuda=False):
        self.img_transform = img_transform
        super(CaptionGenerator, self).__init__(model,
                                               None,
                                               target_tok,
                                               beam_size,
                                               length_normalization_factor,
                                               max_sequence_length,
                                               get_attention,
                                               cuda)

    def set_src_language(self, language):
        pass

    def describe(self, input_img, target_priming=None):
        target_priming = target_priming or ''
        src_img = self.img_transform(input_img)

        bos = self.target_tok.tokenize(
            target_priming, insert_start=self.insert_target_start)
        src = Variable(src_img.unsqueeze(0).unsqueeze(0), volatile=True)
        if self.cuda:
            src = src.cuda()
        if target_priming is None:
            bos = self.insert_target_start
        else:
            bos = list(self.target_tok.tokenize(target_priming,
                                                insert_start=self.insert_target_start))
        self.model.clear_state()
        context = self.model.encode(src)
        _, c, h, w = list(context[0].size())
        if hasattr(self.model, 'bridge'):
            context = self.model.bridge(context)

        [preds], [logprobs], [attentions] = self.generator.beam_search([bos], [
                                                                       context])
        # remove forced  tokens
        output = self.target_tok.detokenize(
            preds[len(self.insert_target_start):-1])
        if len(target_priming) > 0:
            output = [' '.join([target_priming, o]) for o in output]
        if attentions is not None:
            attentions = torch.stack([a.view(h, w) for a in attentions], 0)
            preds = preds[len(self.insert_target_start):]
            preds = [self.target_tok.idx2word(idx) for idx in preds]

        return output, (attentions, preds)
        # for s,p in zip(sentences,logprob):
        #     print(target_tok.detokenize(s)[::-1],' p=%s' % math.exp(p))

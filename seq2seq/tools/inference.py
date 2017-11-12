#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from .config import EOS, BOS, LANGUAGE_TOKENS
from .beam_search import SequenceGenerator
from torch.nn.functional import adaptive_avg_pool2d
from seq2seq import models
from seq2seq.tools import batch_sequences


def average_models(checkpoint_filenames):
    averaged = {}
    scale = 1. / len(checkpoint_filenames)
    print('Averaging %s models' % len(checkpoint_filenames))
    for m in checkpoint_filenames:
        checkpoint = torch.load(
            m, map_location=lambda storage, loc: storage)
        for n, p in checkpoint['state_dict'].items():
            if n in averaged:
                averaged[n].add_(scale * p)
            else:
                averaged[n] = scale * p
    checkpoint['state_dict'] = averaged
    return checkpoint


class Translator(object):

    def __init__(self, checkpoint=None,
                 model=None, src_tok=None, target_tok=None,
                 beam_size=5,
                 length_normalization_factor=0,
                 max_sequence_length=50,
                 get_attention=False,
                 cuda=None):
        assert checkpoint is not None or \
            model is not None and src_tok is not None and target_tok is not None, \
            "supply either a checkpoint dictionary or model and tokenizers"
        if checkpoint is not None:
            config = checkpoint['config']
            self.model = getattr(models, config.model)(**config.model_config)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.src_tok, self.target_tok = checkpoint['tokenizers'].values()
        else:
            self.model = model
            self.src_tok = src_tok
            self.target_tok = target_tok
        self.insert_target_start = [BOS]
        self.insert_src_start = [BOS]
        self.insert_src_end = [EOS]
        self.get_attention = get_attention
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        if self.cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.eval()
        self.generator = SequenceGenerator(
            model=self.model,
            beam_size=beam_size,
            max_sequence_length=max_sequence_length,
            get_attention=get_attention,
            length_normalization_factor=length_normalization_factor)

    def set_src_language(self, language=None):
        if language is None:
            self.insert_src_start = [BOS]
        else:
            lang = self.src_tok.special_tokens.index(LANGUAGE_TOKENS(language))
            self.insert_src_start = [lang]

    def set_target_language(self, language=None):
        if language is None:
            self.insert_target_start = [BOS]
        else:
            lang = self.target_tok.special_tokens.index(
                LANGUAGE_TOKENS(language))
            self.insert_target_start = [lang]

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

        src = Variable(batch_sequences(src_tok,
                                       batch_first=self.model.encoder.batch_first)[0],
                       volatile=True)
        if self.cuda:
            src = src.cuda()

        context = self.model.encode(src)
        if hasattr(self.model, 'bridge'):
            state = self.model.bridge(context)
        state_list = [state[i] for i in range(batch)]

        seqs = self.generator.beam_search(bos, state_list)
        # remove forced  tokens
        preds = [s.sentence[len(self.insert_target_start):] for s in seqs]
        output = [self.target_tok.detokenize(p[:-1]) for p in preds]

        output = output[0] if flatten else output
        if self.get_attention:
            attentions = [s.attention for s in seqs]
            # if target_priming is not None:
            # preds = [preds[b][-len(attentions[b]):] for b in range(batch)]
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
        state = self.model.encode(src)
        _, c, h, w = list(state.outputs.size())
        if hasattr(self.model, 'bridge'):
            state = self.model.bridge(state)

        [seq] = self.generator.beam_search([bos], [state])
        # remove forced  tokens
        output = self.target_tok.detokenize(
            seq.sentence[len(self.insert_target_start):-1])
        if len(target_priming) > 0:
            output = [' '.join([target_priming, o]) for o in output]
        if seq.attention is not None:
            attentions = torch.stack([a.view(h, w) for a in seq.attention], 0)
            preds = seq.sentence[len(self.insert_target_start):]
            preds = [self.target_tok.idx2word(idx) for idx in preds]
            return output, (attentions, preds)
        else:
            return output

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .config import EOS, BOS, LANGUAGE_TOKENS
from .beam_search import SequenceGenerator
from torch.nn.functional import adaptive_avg_pool2d
from seq2seq import models
from seq2seq.tools import batch_sequences
from seq2seq.models.modules.weight_norm import WeightNorm
from torch.nn.utils.rnn import PackedSequence


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


def remove_wn_checkpoint(checkpoint):
    model = getattr(models, checkpoint['config'].model)(
        **checkpoint['config'].model_config)
    model.load_state_dict(checkpoint['state_dict'])

    def change_field(dict_obj, field, new_val):
        for k, v in dict_obj.items():
            if k == field:
                dict_obj[k] = new_val
            elif isinstance(v, dict):
                change_field(v, field, new_val)

    change_field(checkpoint['config'].model_config, 'layer_norm', False)

    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.LSTM) or isinstance(module, nn.LSTMCell):
            for n, _ in list(module.named_parameters()):
                if n.endswith('_g'):
                    name = n.replace('_g', '')
                    wn = WeightNorm(None, 0)
                    weight = wn.compute_weight(module, name)
                    delattr(module, name)
                    del module._parameters[name + '_g']
                    del module._parameters[name + '_v']
                    module.register_parameter(name, nn.Parameter(weight.data))
                    print('wn removed from %s - %s' % (module, name))

    checkpoint['state_dict'] = model.state_dict()
    change_field(checkpoint['config'].model_config, 'weight_norm', False)
    return checkpoint


class Translator(object):

    def __init__(self, checkpoint,
                 use_moses=None,
                 beam_size=5,
                 length_normalization_factor=0,
                 max_input_length=None,
                 max_output_length=50,
                 get_attention=False,
                 device="cpu",
                 device_ids=None):
        config = checkpoint['config']
        self.model = getattr(models, config.model)(**config.model_config)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.src_tok, self.target_tok = checkpoint['tokenizers'].values()
        if use_moses is None:  # if not set, turn on if training was done with moses pretok
            use_moses = config.data_config.get('moses_pretok', False)
        if use_moses:
            src_lang, target_lang = checkpoint['tokenizers'].keys()
            self.src_tok.enable_moses(lang=src_lang)
            self.target_tok.enable_moses(lang=target_lang)
        self.insert_target_start = [BOS]
        self.insert_src_start = [BOS]
        self.insert_src_end = [EOS]
        self.get_attention = get_attention
        self.device = device
        self.device_ids = device_ids
        self.model.to(self.device)
        self.model.eval()

        self.beam_size = beam_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.get_attention = get_attention
        self.length_normalization_factor = length_normalization_factor
        self.batch_first = self.model.encoder.batch_first
        self.pack_encoder_inputs = getattr(self.model.encoder, 'pack_inputs',
                                           False)

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

        order = range(batch)
        if self.pack_encoder_inputs:
            # sort by the first set
            sorted_idx, src_tok = zip(*sorted(
                enumerate(src_tok), key=lambda x: x[1].numel(), reverse=True))
            order = [sorted_idx.index(i) for i in order]

        if target_priming is None:
            bos = [self.insert_target_start] * batch
        else:
            if isinstance(target_priming, list):
                bos = [list(self.target_tok.tokenize(target_priming[i],
                                                     insert_start=self.insert_target_start))
                       for i in order]
            else:
                bos = self.target_tok.tokenize(target_priming,
                                               insert_start=self.insert_target_start)
                bos = [list(bos)] * batch

        src = batch_sequences(src_tok,
                              max_length=self.max_input_length,
                              sort=False,
                              pack=self.pack_encoder_inputs,
                              device=self.device,
                              batch_first=self.batch_first)[0]

        with torch.no_grad():
            seqs = self.model.generate(src, bos,
                                       beam_size=self.beam_size,
                                       max_sequence_length=self.max_output_length,
                                       length_normalization_factor=self.length_normalization_factor,
                                       get_attention=self.get_attention, device_ids=self.device_ids)
        # remove forced  tokens
        preds = [s.output[len(self.insert_target_start):] for s in seqs]
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

    def __init__(self, checkpoint=None,
                 image_transform=None,
                 use_moses=False,
                 beam_size=5,
                 length_normalization_factor=0,
                 max_output_length=50,
                 get_attention=False,
                 device="cpu"):
        super(CaptionGenerator, self).__init__(checkpoint=checkpoint,
                                               use_moses=use_moses,
                                               beam_size=beam_size,
                                               length_normalization_factor=length_normalization_factor,
                                               max_output_length=max_output_length,
                                               get_attention=get_attention, device=device)
        self.image_transform = image_transform or self.src_tok(
            allow_var_size=False, train=False)

    def set_src_language(self, language):
        pass

    def encode(self, input_image):
        with torch.no_grad():
            src_img = self.image_transform(input_image)
            src = src_img.unsqueeze(0).unsqueeze(0).to(self.device)
            state = self.model.encode(src)
            if hasattr(self.model, 'bridge'):
                state = self.model.bridge(state)

    def describe(self, input_image, target_priming=None):
        with torch.no_grad():
            target_priming = target_priming or ''

            bos = self.target_tok.tokenize(
                target_priming, insert_start=self.insert_target_start)
            if target_priming is None:
                bos = self.insert_target_start
            else:
                bos = list(self.target_tok.tokenize(target_priming,
                                                    insert_start=self.insert_target_start))
            src_img = self.image_transform(input_image)
            src_img = src_img.unsqueeze(0).unsqueeze(0).to(self.device)
            seq = self.model.generate(src_img, [bos],
                                      beam_size=self.beam_size,
                                      max_sequence_length=self.max_output_length,
                                      length_normalization_factor=self.length_normalization_factor,
                                      get_attention=self.get_attention)
            # remove forced  tokens
            preds = [s.output[len(self.insert_target_start):] for s in seq]
            output = [self.target_tok.detokenize(p[:-1]) for p in preds]
            if len(target_priming) > 0:
                output = [' '.join([target_priming, o]) for o in output]
            if self.get_attention and seq.attention is not None:
                _, c, h, w = list(state.outputs.size())
                attentions = torch.stack([a.view(h, w)
                                          for a in seq.attention], 0)
                preds = seq.output[len(self.insert_target_start):]
                preds = [self.target_tok.idx2word(idx) for idx in preds]
                return output, (attentions, preds)
            else:
                return output

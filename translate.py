#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import codecs
import torch
from tools.translator import Translator
parser = argparse.ArgumentParser(
    description='Translate a file using pretrained model')

parser.add_argument('input', help='input file for translation')
parser.add_argument('-o', '--output', help='output file')
parser.add_argument('-m', '--model', default='./results/recurrent_attention_wmt16/checkpoint.pth.tar',
                    help='model checkpoint file')
parser.add_argument('--beam_size', default=5, type=int,
                    help='beam size used')


if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint = torch.load(args.model)
    model = checkpoint['model']
    src_tok, target_tok = checkpoint['tokenizers'].values()
    cuda = True
    translation_model = Translator(model,
                                   src_tok=src_tok,
                                   target_tok=target_tok,
                                   beam_size=5,
                                   length_normalization_factor=0,
                                   cuda=cuda)

    output_file = codecs.open(args.output, 'w', encoding='UTF-8')
    with codecs.open(args.input, encoding='UTF-8') as input_file:
        for line in input_file:
            translated = translation_model.translate(line)
            output_file.write(translated)
            output_file.write('\n')
    output_file.close()

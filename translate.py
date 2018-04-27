#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
from ast import literal_eval
import torch
from seq2seq.tools.inference import Translator, average_models

parser = argparse.ArgumentParser(
    description='Translate a file using pretrained model')

parser.add_argument('input', help='input file for translation')
parser.add_argument('-o', '--output', help='output file')
parser.add_argument('-m', '--model', help='model checkpoint file')
parser.add_argument('--beam_size', default=8, type=int,
                    help='beam size used')
parser.add_argument('--max_sequence_length', default=50, type=int,
                    help='maximum prediciton length')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size used for inference')
parser.add_argument('--length_normalization', default=0.6, type=float,
                    help='length normalization factor')
parser.add_argument('--devices', default='0',
                    help='device assignment (e.g "0,1", {"encoder":0, "decoder":1})')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--verbose', action='store_true',
                    help='print translations on screen')

if __name__ == '__main__':
    args = parser.parse_args()
    args.devices = literal_eval(args.devices)
    try:
        args.model = literal_eval(args.model)
    except:
        pass
    if 'cuda' in args.type and torch.cuda.is_available():
        device = "cuda"
        main_gpu = 0
        if isinstance(args.devices, tuple):
            main_gpu = args.devices[0]
        elif isinstance(args.devices, int):
            main_gpu = args.devices
        elif isinstance(args.devices, dict):
            main_gpu = args.devices.get('input', 0)
        torch.cuda.set_device(main_gpu)
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    if isinstance(args.model, tuple):  # average models
        checkpoint = average_models(args.model)
    else:
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)

    translation_model = Translator(checkpoint=checkpoint,
                                   beam_size=args.beam_size,
                                   max_sequence_length=args.max_sequence_length,
                                   length_normalization_factor=args.length_normalization,
                                   device=device)

    output_file = codecs.open(args.output, 'w', encoding='UTF-8')

    def write_output(lines, source=None):
        if args.verbose:
            for i in range(len(lines)):
                print('\n SOURCE:\t %s TRANSLATION:\t %s' %
                      (source[i], lines[i]))
        for l in lines:
            output_file.write(l)
            output_file.write('\n')

    with codecs.open(args.input, encoding='UTF-8') as input_file:
        lines = []
        for line in input_file:
            if len(lines) < args.batch_size:
                lines.append(line)
                continue
            else:
                write_output(translation_model.translate(lines), lines)
                lines = [line]
        if len(lines) > 0:
            write_output(translation_model.translate(lines), lines)

    output_file.close()

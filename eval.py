#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import logging
from ast import literal_eval
from datetime import datetime
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from seq2seq import models, datasets
from seq2seq.tools.utils.log import setup_logging
from seq2seq.tools.config import PAD
import seq2seq.tools.trainer as trainers


parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')
parser.add_argument('--checkpoint', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--dataset', metavar='DATASET', default='WMT16_de_en',
                    choices=datasets.__all__,
                    help='dataset used: ' +
                    ' | '.join(datasets.__all__) +
                    ' (default: WMT16_de_en)')
parser.add_argument('--dataset_dir', metavar='DATASET_DIR',
                    help='dataset dir')
parser.add_argument('--data_config',
                    default="{'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}",
                    help='data configuration')
parser.add_argument('--devices', default='0',
                    help='device assignment (e.g "0,1", {"encoder":0, "decoder":1})')
parser.add_argument('--trainer', metavar='TRAINER', default='Seq2SeqTrainer',
                    choices=trainers.__all__,
                    help='trainer used: ' +
                    ' | '.join(trainers.__all__) +
                    ' (default: Seq2SeqTrainer)')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--pack_encoder_inputs', action='store_true',
                    help='pack encoder inputs for rnns')
parser.add_argument('--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--max_length', default=100, type=int,
                    help='maximum sequence length')
parser.add_argument('--max_tokens', default=None, type=int,
                    help='maximum sequence tokens')


def main(args):
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join('/tmp', time_stamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logging(os.path.join(save_path, 'log.txt'))

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    args.devices = literal_eval(args.devices)
    if 'cuda' in args.type:
        main_gpu = 0
        if isinstance(args.devices, tuple):
            main_gpu = args.devices[0]
        elif isinstance(args.devices, int):
            main_gpu = args.devices
        elif isinstance(args.devices, dict):
            main_gpu = args.devices.get('input', 0)
        torch.cuda.set_device(main_gpu)
        cudnn.benchmark = True


    checkpoint = torch.load(args.checkpoint , map_location=lambda storage, loc: storage)
    config = checkpoint['config']
    src_tok, target_tok = checkpoint['tokenizers'].values()

    args.data_config = literal_eval(args.data_config)
    dataset = getattr(datasets, args.dataset)
    args.data_config['tokenizers'] = checkpoint['tokenizers']
    val_data = dataset(args.dataset_dir, split='dev', **args.data_config)

    model = getattr(models, config.model)(**config.model_config)
    model.load_state_dict(checkpoint['state_dict'])

    batch_first = getattr(model, 'batch_first', False)

    logging.info(model)

    # define data loaders
    val_loader = val_data.get_loader(batch_size=args.batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     augment=False,
                                     pack=args.pack_encoder_inputs,
                                     max_length=args.max_length,
                                     max_tokens=args.max_tokens,
                                     num_workers=args.workers)

    trainer_options = dict(
        save_path=save_path,
        devices=args.devices,
        print_freq=args.print_freq)

    trainer_options['model'] = model
    trainer = getattr(trainers, args.trainer)(**trainer_options)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    model.type(args.type)

    trainer.evaluate(val_loader)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

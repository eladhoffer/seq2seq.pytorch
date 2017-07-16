#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import logging
from ast import literal_eval
from datetime import datetime
from math import inf
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from seq2seq import models, datasets
from seq2seq.tools.utils import setup_logging, ResultsLog
from seq2seq.tools.config import PAD
import seq2seq.tools.trainer as trainers


parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')
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
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--model', metavar='MODEL', default='RecurrentAttentionSeq2Seq',
                    choices=models.__all__,
                    help='model architecture: ' +
                    ' | '.join(models.__all__) +
                    ' (default: RecurrentAttentionSeq2Seq)')
parser.add_argument('--model_config', default="{'hidden_size:256','num_layers':2}",
                    help='architecture configuration')
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
parser.add_argument('--epochs', default=90, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--optimization_config',
                    default="{0: {'optimizer': SGD, 'lr':0.1, 'momentum':0.9}}",
                    type=str, metavar='OPT',
                    help='optimization regime used')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--grad_clip', default=5., type=float,
                    help='maximum grad norm value')
parser.add_argument('--max_length', default=100, type=int,
                    help='maximum sequence length')


def main(args):
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

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

    dataset = getattr(datasets, args.dataset)
    args.data_config = literal_eval(args.data_config)
    train_data = dataset(args.dataset_dir, split='train', **args.data_config)
    val_data = dataset(args.dataset_dir, split='dev', **args.data_config)
    src_tok, target_tok = train_data.tokenizers.values()

    regime = literal_eval(args.optimization_config)
    model_config = literal_eval(args.model_config)

    model_config.setdefault('encoder', {})
    model_config.setdefault('decoder', {})
    model_config['encoder']['vocab_size'] = src_tok.vocab_size()
    model_config['decoder']['vocab_size'] = target_tok.vocab_size()
    model_config['vocab_size'] = target_tok.vocab_size()
    args.model_config = model_config

    model = getattr(models, args.model)(**model_config)
    batch_first = getattr(model, 'batch_first', False)

    logging.info(model)

    # define data loaders
    train_loader = train_data.get_loader(batch_size=args.batch_size,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         max_length=args.max_length,
                                         num_workers=args.workers)
    val_loader = val_data.get_loader(batch_size=args.batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     max_length=args.max_length,
                                     num_workers=args.workers)
    # define loss function (criterion) and optimizer
    loss_weight = torch.ones(target_tok.vocab_size())
    loss_weight[PAD] = 0
    criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    criterion.type(args.type)

    trainer_options = dict(
        criterion=criterion,
        grad_clip=args.grad_clip,
        save_path=save_path,
        save_info={'tokenizers': train_data.tokenizers,
                   'config': args},
        regime=regime,
        batch_first=batch_first,
        devices=args.devices,
        print_freq=args.print_freq)

    trainer_options['model'] = model
    trainer = getattr(trainers, args.trainer)(**trainer_options)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    model.type(args.type)

    # optionally resume from a checkpoint
    if args.evaluate:
        trainer.load(args.evaluate)
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            trainer.load(checkpoint_file)
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    logging.info('training regime: %s', regime)

    best_perplexity = inf
    for epoch in range(args.start_epoch, args.epochs):
        trainer.epoch = epoch
        # train for one epoch
        train_loss, train_perplexity = trainer.optimize(train_loader)

        # evaluate on validation set
        val_loss, val_perplexity = trainer.evaluate(val_loader)

        # remember best prec@1 and save checkpoint
        is_best = val_perplexity < best_perplexity
        best_perplexity = min(val_perplexity, best_perplexity)
        if is_best:
            trainer.save(is_best=True)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Training Perplexity {train_perplexity:.4f} \t'
                     'Validation Perplexity {val_perplexity:.4f} \t'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_perplexity=train_perplexity, val_perplexity=val_perplexity))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_perplexity=train_perplexity, val_perplexity=val_perplexity)
        results.plot(x='epoch', y=['train_loss', 'val_loss'],
                     title='Loss', ylabel='loss')
        results.plot(x='epoch', y=['train_perplexity', 'val_perplexity'],
                     title='Perplexity', ylabel='perplexity')

        results.save()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

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
import models
import datasets
from tools.utils import setup_logging, ResultsLog
import tools.trainer as trainers
from tools.config import PAD
from tools.translator import Translator

Datasets = ['WMT16_de_en', 'OpenSubtitles2016']
Models = ['Transformer', 'RecurrentAttentionSeq2Seq', 'GNMT']
Trainers = ['MultiSeq2SeqTrainer', 'Seq2SeqTrainer']

parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')
parser.add_argument('--dataset', metavar='DATASET', default='WMT16_de_en',
                    choices=Datasets,
                    help='dataset used: ' +
                    ' | '.join(Datasets) +
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
                    choices=Models,
                    help='model architecture: ' +
                    ' | '.join(Models) +
                    ' (default: RecurrentAttentionSeq2Seq)')
parser.add_argument('--model_config', default="{'hidden_size:256','num_layers':2}",
                    help='architecture configuration')

parser.add_argument('--trainer', metavar='TRAINER', default='Seq2SeqTrainer',
                    choices=Trainers,
                    help='trainer used: ' +
                    ' | '.join(Trainers) +
                    ' (default: Seq2SeqTrainer)')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
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

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    data_config = dict(root=args.dataset_dir)
    if data_config is not '':
        data_config = dict(data_config, **literal_eval(args.data_config))
    dataset = getattr(datasets, args.dataset)
    train_data = dataset(split='train', **data_config)
    val_data = dataset(split='dev', **data_config)
    _, target_tok = train_data.tokenizers.values()


    regime = literal_eval(args.optimization_config)

    regime = literal_eval(args.optimization_config)
    model_config = dict(vocab_size=target_tok.vocab_size(),
                        **literal_eval(args.model_config))
    model = getattr(models, args.model)(**model_config)
    batch_first = getattr(model, 'batch_first', False)

    logging.info(model)

    #define data loaders
    train_loader = train_data.get_loader(batch_size=args.batch_size,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         num_workers=args.workers)
    val_loader = val_data.get_loader(batch_size=args.batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
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
        gpus=args.gpus,
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

    best_perplexity = 0
    for epoch in range(args.start_epoch, args.epochs):
        trainer.epoch = epoch
        # train for one epoch
        train_loss, train_perplexity = trainer.optimize(train_loader)

        # evaluate on validation set
        val_loss, val_perplexity = trainer.evaluate(val_loader)
        #
        # translation_model = Translator(model,
        #                                src_tok=src_tok,
        #                                target_tok=target_tok,
        #                                beam_size=5,
        #                                length_normalization_factor=0,
        #                                cuda=True)
        # for i in range(10):
        #     src_seq, target_seq = val_data[i]
        #     src_seq = src_tok.detokenize(src_seq[1:-1])
        #     target_seq = target_tok.detokenize(target_seq[1:-1])
        #     pred = translation_model.translate(src_seq)
        #     logging.info('\n Example %s:'
        #                  '\n \t Source: %s'
        #                  '\n \t Target: %s'
        #                  '\n \t Prediction: %s'
        #                  % (i, src_seq, target_seq, pred))

        # remember best prec@1 and save checkpoint
        is_best = val_perplexity > best_perplexity
        best_perplexity = max(val_perplexity, best_perplexity)
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

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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from seq2seq import models, datasets
from seq2seq.tools.utils.log import setup_logging
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
parser.add_argument('--device_ids', default='0',
                    help='device ids assignment (e.g "0,1", {"encoder":0, "decoder":1})')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--trainer', metavar='TRAINER', default='Seq2SeqTrainer',
                    choices=trainers.__all__,
                    help='trainer used: ' +
                    ' | '.join(trainers.__all__) +
                    ' (default: Seq2SeqTrainer)')
parser.add_argument('--dtype', default='torch.float',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='rank of distributed processes')
parser.add_argument('--dist-init', default='env://', type=str,
                    help='init used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--optimization_config',
                    default="[{'epoch':0, 'optimizer':'SGD', 'lr':0.1, 'momentum':0.9}]",
                    type=str, metavar='OPT',
                    help='optimization regime used')
parser.add_argument('--print-freq', default=50, type=int,
                    help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=1000, type=int,
                    help='save frequency (default: 1000)')
parser.add_argument('--eval-freq', default=2500, type=int,
                    help='evaluation frequency (default: 2500)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--grad_clip', default='5.', type=str,
                    help='maximum grad norm value')
parser.add_argument('--embedding_grad_clip', default=None, type=float,
                    help='maximum embedding grad norm value')
parser.add_argument('--label_smoothing', default=0, type=float,
                    help='label smoothing coefficient - default 0')
parser.add_argument('--uniform_init', default=None, type=float,
                    help='if value not None - init weights to U(-value,value)')
parser.add_argument('--max_length', default=100, type=int,
                    help='maximum sequence length')
parser.add_argument('--max_tokens', default=None, type=int,
                    help='maximum sequence tokens')
parser.add_argument('--limit_num_tokens', default=None, type=int,
                    help='trim batch size to fit maximum num of tokens')


def main(args):
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.distributed = args.local_rank >= 0 or args.world_size > 1
    setup_logging(os.path.join(save_path, 'log.txt'),
                  dummy=args.distributed and args.local_rank > 0)

    if args.distributed:
        args.device_ids = args.local_rank
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init,
                                world_size=args.world_size, rank=args.local_rank)
    else:
        args.device_ids = literal_eval(args.device_ids)

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    device = args.device
    if 'cuda' in args.device:
        main_gpu = 0
        if isinstance(args.device_ids, tuple):
            main_gpu = args.device_ids[0]
        elif isinstance(args.device_ids, int):
            main_gpu = args.device_ids
        elif isinstance(args.device_ids, dict):
            main_gpu = args.device_ids.get('input', 0)
        torch.cuda.set_device(main_gpu)
        cudnn.benchmark = True
        device = torch.device(device, main_gpu)

    dataset = getattr(datasets, args.dataset)
    args.data_config = literal_eval(args.data_config)
    args.grad_clip = literal_eval(args.grad_clip)
    train_data = dataset(args.dataset_dir, split='train', **args.data_config)
    val_data = dataset(args.dataset_dir, split='dev', **args.data_config)
    src_tok, target_tok = train_data.tokenizers.values()

    regime = literal_eval(args.optimization_config)
    model_config = literal_eval(args.model_config)

    model_config.setdefault('encoder', {})
    model_config.setdefault('decoder', {})
    if hasattr(src_tok, 'vocab_size'):
        model_config['encoder']['vocab_size'] = src_tok.vocab_size
    model_config['decoder']['vocab_size'] = target_tok.vocab_size
    model_config['vocab_size'] = model_config['decoder']['vocab_size']
    args.model_config = model_config

    model = getattr(models, args.model)(**model_config)

    model.to(device)
    batch_first = getattr(model, 'batch_first', False)

    logging.info(model)
    pack_encoder_inputs = getattr(model.encoder, 'pack_inputs', False)

    # define data loaders
    if args.distributed:
        train_sampler = DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = train_data.get_loader(batch_size=args.batch_size,
                                         batch_first=batch_first,
                                         shuffle=train_sampler is None,
                                         augment=True,
                                         sampler=train_sampler,
                                         pack=pack_encoder_inputs,
                                         max_length=args.max_length,
                                         max_tokens=args.max_tokens,
                                         num_workers=args.workers,
                                         drop_last=True)
    val_loader = val_data.get_loader(batch_size=args.batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     augment=False,
                                     pack=pack_encoder_inputs,
                                     max_length=args.max_length,
                                     max_tokens=args.max_tokens,
                                     num_workers=args.workers)

    trainer_options = dict(
        grad_clip=args.grad_clip,
        embedding_grad_clip=args.embedding_grad_clip,
        label_smoothing=args.label_smoothing,
        save_path=save_path,
        save_info={'tokenizers': train_data.tokenizers,
                   'config': args},
        regime=regime,
        limit_num_tokens=args.limit_num_tokens,
        distributed=args.distributed,
        local_rank=args.local_rank,
        device_ids=args.device_ids,
        device=device,
        dtype=args.dtype,
        print_freq=args.print_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq)

    trainer_options['model'] = model
    trainer = getattr(trainers, args.trainer)(**trainer_options)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    if args.uniform_init is not None:
        for param in model.parameters():
            param.data.uniform_(args.uniform_init, -args.uniform_init)

    # optionally resume from a checkpoint
    if args.evaluate:
        trainer.load(args.evaluate)
        trainer.evaluate(val_loader)
        return
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            trainer.load(checkpoint_file)
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    logging.info('training regime: %s', regime)
    trainer.epoch = args.start_epoch

    while trainer.epoch < args.epochs:
        # train for one epoch
        trainer.run(train_loader, val_loader)
        print('end1')
    print('end2')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

# -*- coding: utf-8 -*-
import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
from datetime import datetime
from models.recurrent import RecurentEncoder, RecurentDecoder
from models.gnmt import GNMT
from models.conv import ConvSeq2Seq
from models.seq2seq import Seq2Seq
from tools.utils import *
from tools.trainer import Seq2SeqTrainer
from datasets import MultiLanguageDataset, WMT16_de_en
from tools.config import *

parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--grad_clip', default=5., type=float,
                    help='maximum grad norm value')


def main():
    global args, best_perplexity
    best_perplexity = 0
    args = parser.parse_args()

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

    # Data loading code
    train_data = WMT16_de_en(
        root='/media/drive/Datasets/wmt16_de_en', split='train')
    val_data = WMT16_de_en(
        root='/media/drive/Datasets/wmt16_de_en', split='dev')
    src_tok, target_tok = train_data.tokenizers.values()

    train_loader = train_data.get_loader(batch_size=args.batch_size, batch_first=True,
                                         shuffle=True, num_workers=args.workers)
    val_loader = val_data.get_loader(batch_size=args.batch_size, batch_first=True,
                                     shuffle=False, num_workers=args.workers)
    regime = {e: {'optimizer': args.optimizer,
                  'lr': args.lr * (0.5 ** e),
                  'momentum': args.momentum,
                  'weight_decay': args.weight_decay} for e in range(10)}

    # define loss function (criterion) and optimizer
    loss_weight = torch.ones(target_tok.vocab_size())
    loss_weight[PAD] = 0
    criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    criterion.type(args.type)

    # model = Seq2Seq(encoder=encoder, decoder=decoder)
    model = ConvSeq2Seq(target_tok.vocab_size())
    trainer = Seq2SeqTrainer(model,
                             criterion=criterion,
                             optimizer=torch.optim.SGD,
                             grad_clip=args.grad_clip,
                             regime=regime,
                             batch_first=True,
                             print_freq=args.print_freq)
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
            trainer.load(args.evaluate)
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    logging.info('training regime: %s', regime)

    for epoch in range(args.start_epoch, args.epochs):


        model.eval()
        for i in range(10):
            src_seq, target_seq = val_data[i]
            src_seq = Variable(src_seq.cuda(), volatile=True)
            target_seq = Variable(target_seq.cuda(), volatile=True)
            pred_seq = model(src_seq.unsqueeze(
                0), target_seq[:-1].unsqueeze(0))
            _, pred_seq = pred_seq.max(2)
            pred_seq = pred_seq.view(-1)
            logging.info('\n Example {0}:'
                         '\n \t Source: {src}'
                         '\n \t Target: {target}'
                         '\n \t Prediction: {pred}'
                         .format(i,
                                 src=src_tok.detokenize(src_seq.data[1:]),
                                 target=target_tok.detokenize(
                                     target_seq[1:].data),
                                 pred=target_tok.detokenize(pred_seq.data[:])))
        # train for one epoch
        train_loss, train_perplexity = trainer.optimize(train_loader)

        # evaluate on validation set
        val_loss, val_perplexity = trainer.evaluate(val_loader)

        # remember best prec@1 and save checkpoint
        is_best = val_perplexity < best_perplexity
        best_perplexity = min(val_perplexity, best_perplexity)
        if is_best:
            model.save(path=save_path)

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
    main()

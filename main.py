import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from utils import *
from datetime import datetime
from ast import literal_eval
from models.recurrent import RecurentEncoder, RecurentDecoder
from seq2seq import Seq2Seq
from datasets import MultiLanguageDataset
from config import *
import pdb

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
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
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
    languages = ['en', 'he']
    data = MultiLanguageDataset(languages=languages)
    src_tok, target_tok = data.tokenizers.values()
    train_data, val_data = data, data#data.split(0.1)

    encoder = RecurentEncoder(src_tok.vocab_size(),
                              hidden_size=128, num_layers=1, bidirectional=True)
    decoder = RecurentDecoder(target_tok.vocab_size(),
                              hidden_size=128, num_layers=2)

    train_loader = train_data.get_loader(batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers)
    val_loader = val_data.get_loader(batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)
    regime = {e: {'optimizer': args.optimizer,
                  'lr': args.lr * (0.1 ** e),
                  'momentum': args.momentum,
                  'weight_decay': args.weight_decay} for e in range(10)}
    # define loss function (criterion) and optimizer
    loss_weight = torch.ones(target_tok.vocab_size())
    loss_weight[PAD] = 0
    criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    criterion.type(args.type)

    model = Seq2Seq(encoder=encoder,
                    decoder=decoder,
                    criterion=criterion,
                    optimizer=torch.optim.SGD,
                    regime=regime)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    model.type(args.type)

    # optionally resume from a checkpoint
    if args.evaluate:
        model.load(args.evaluate)
    # elif args.resume:
    #     checkpoint_file = args.resume
    #     if os.path.isdir(checkpoint_file):
    #         results.load(os.path.join(checkpoint_file, 'results.csv'))
    #         checkpoint_file = os.path.join(
    #             checkpoint_file, 'model_best.pth.tar')
    #     if os.path.isfile(checkpoint_file):
    #         logging.info("loading checkpoint '%s'", args.resume)
    #         checkpoint = torch.load(checkpoint_file)
    #         args.start_epoch = checkpoint['epoch']
    #         best_perplexity = checkpoint['best_perplexity']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         logging.info("loaded checkpoint '%s' (epoch %s)",
    #                      checkpoint_file, checkpoint['epoch'])
    #     else:
    #         logging.error("no checkpoint found at '%s'", args.resume)

    logging.info('training regime: %s', regime)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_loss, train_perplexity = model.optimize(train_loader)

        # evaluate on validation set
        val_loss, val_perplexity = model.evaluate(val_loader)
        #
        # model.eval()
        # for i in range(10):
        #     src_seq = Variable(val_src_data[i].cuda(), volatile=True)
        #     target_seq = Variable(val_target_data[i].cuda(), volatile=True)
        #     pred_seq = model(src_seq.unsqueeze(
        #         1), target_seq[:-1].unsqueeze(1))
        #     _, pred_seq = pred_seq.max(2)
        #     pred_seq = pred_seq.view(-1)
        #     logging.info('\n Example {0}:'
        #                  '\n \t Source: {src}'
        #                  '\n \t Target: {target}'
        #                  '\n \t Prediction: {pred}'
        #                  .format(i,
        #                          src=src_tok.detokenize(src_seq.data[1:]),
        #                          target=target_tok.detokenize(
        #                              target_seq[1:].data),
        #                          pred=target_tok.detokenize(pred_seq.data[:])))

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

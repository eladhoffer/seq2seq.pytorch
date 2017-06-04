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
from data import AlignedDatasets, NarrowDataset, create_padded_batch, get_dataset_bpe
from models.recurrent import RecurentEncoder, RecurentDecoder
from models.seq2seq import Seq2Seq
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
    [src_data, target_data], [src_tok, target_tok] = get_dataset_bpe(
        langs=['en', 'he'])
    size_train = int(len(src_data) * 0.9)
    train_src_data = NarrowDataset(src_data, 0, size_train - 1)
    val_src_data = NarrowDataset(src_data, size_train, None)
    train_target_data = NarrowDataset(target_data, 0, size_train - 1)
    val_target_data = NarrowDataset(target_data, size_train, None)
    # train_src_data = NarrowDataset(src_data, 0, 10000)  # size_train - 1)
    # val_src_data = NarrowDataset(src_data, 2000, 2100)  # size_train, None)
    # train_target_data = NarrowDataset(target_data, 0, 10000)  # size_train - 1)
    # val_target_data = NarrowDataset(
    #     target_data, 2000, 2100)  # size_train, None)
    # create model
    encoder = RecurentEncoder(src_tok.vocab_size(),
                              hidden_size=128, num_layers=1, bidirectional=True)
    decoder = RecurentDecoder(target_tok.vocab_size(),
                              hidden_size=128, num_layers=2)
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    train_loader = torch.utils.data.DataLoader(AlignedDatasets([train_src_data, train_target_data]),
                                               batch_size=args.batch_size,
                                               collate_fn=create_padded_batch(),
                                               shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(AlignedDatasets([val_src_data, val_target_data]),
                                             batch_size=args.batch_size,
                                             collate_fn=create_padded_batch(),
                                             num_workers=args.workers)
    regime = {e: {'optimizer': args.optimizer,
                  'lr': args.lr * (0.1 ** e),
                  'momentum': args.momentum,
                  'weight_decay': args.weight_decay} for e in range(10)}
    # define loss function (criterion) and optimizer
    loss_weight = torch.ones(target_tok.vocab_size())
    loss_weight[PAD] = 0
    criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    criterion.type(args.type)
    model.type(args.type)
    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch']
            best_perplexity = checkpoint['best_perplexity']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)

    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_loss, train_perplexity = train(
            train_loader, model, criterion, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_perplexity = validate(
            val_loader, model, criterion, epoch)

        model.eval()
        for i in range(10):
            src_seq = Variable(val_src_data[i].cuda(), volatile=True)
            target_seq = Variable(val_target_data[i].cuda(), volatile=True)
            pred_seq = model(src_seq.unsqueeze(1), target_seq[:-1].unsqueeze(1))
            _, pred_seq = pred_seq.max(2)
            pred_seq = pred_seq.view(-1)
            logging.info('\n Example {0}:'
                         '\n \t Source: {src}'
                         '\n \t Target: {target}'
                         '\n \t Prediction: {pred}'
                         .format(i,
                                 src=src_tok.detokenize(src_seq.data[1:]),
                                 target=target_tok.detokenize(target_seq[1:].data),
                                 pred=target_tok.detokenize(pred_seq.data[:])))

        # remember best prec@1 and save checkpoint
        is_best = val_perplexity < best_perplexity
        best_perplexity = min(val_perplexity, best_perplexity)
        save_checkpoint({
            'epoch': epoch + 1,
            'config': args,
            'state_dict': model.state_dict(),
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'best_perplexity': best_perplexity,
            'regime': regime
        }, is_best, path=save_path)
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


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    perplexity = AverageMeter()

    end = time.time()
    for i, ((src, src_length), (target, target_length)) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            src = src.cuda()
            target = target.cuda()
        src_var = Variable(src, volatile=not training)
        target_var = Variable(target, volatile=not training)

        # compute output
        output = model(src_var, target_var[:-1])

        T, B = output.size(0), output.size(1)
        num_words = sum(target_length) - B
        loss = criterion(output.view(T * B, -1).contiguous(),
                         target_var[1:].contiguous().view(-1))
        loss /= num_words
        # measure accuracy and record loss
        losses.update(loss.data[0], num_words)
        perplexity.update(2 ** loss.data[0], num_words)

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Perplexity {perplexity.val:.4f} ({perplexity.avg:.4f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, perplexity=perplexity))

    return losses.avg, perplexity.avg


def translate(model, text, tokenize, detokenize):
    x = tokenize(text)
    y = model(x)
    return detokenize(y)


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

if __name__ == '__main__':
    main()

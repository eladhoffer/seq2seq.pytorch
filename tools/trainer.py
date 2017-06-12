import time
import logging
from itertools import chain
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import shutil
import math
from .utils import *


class Seq2SeqTrainer(object):
    """class for Trainer."""

    def __init__(self, model, criterion,
                 optimizer=None,
                 print_freq=10,
                 save_freq=1000,
                 regime=None,
                 grad_clip=None,
                 batch_first=False,
                 save_info={},
                 save_path='.',
                 cuda=True):
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=0.1)
        self.grad_clip = grad_clip
        self.epoch = 0
        self.save_info = save_info
        self.save_path = save_path
        self.save_freq = save_freq
        self.regime = regime
        self.cuda = cuda
        self.print_freq = print_freq
        self.batch_first = batch_first
        self.lowet_perplexity = None

    def iterate(self, src, target, training=True):
        src, src_length = src
        target, target_length = target
        if self.cuda:
            src = src.cuda()
            target = target.cuda()
        src_var = Variable(src, volatile=not training)
        target_var = Variable(target, volatile=not training)

        # compute output

        if self.batch_first:
            output = self.model(src_var, target_var[:, :-1])
            target_labels = target_var[:, 1:]
        else:
            output = self.model(src_var, target_var[:-1])
            target_labels = target_var[1:]

        T, B = output.size(0), output.size(1)
        num_words = sum(target_length) - B

        loss = self.criterion(output.contiguous().view(T * B, -1),
                              target_labels.contiguous().view(-1))
        loss /= num_words

        if training:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                clip_grad_norm(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        return loss.data[0], num_words

    def feed_data(self, data_loader, training=True):
        if training:
            assert self.optimizer is not None
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        perplexity = AverageMeter()

        end = time.time()
        for i, (src, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # do a train/evaluate iteration
            loss, num_words = self.iterate(src, target, training=training)

            # measure accuracy and record loss
            losses.update(loss, num_words)
            perplexity.update(math.exp(loss), num_words)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Perplexity {perplexity.val:.4f} ({perplexity.avg:.4f})'.format(
                                 self.epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 batch_time=batch_time,
                                 data_time=data_time, loss=losses, perplexity=perplexity))
            if i % self.save_freq == 0:
                self.save()

        return losses.avg, perplexity.avg

    def optimize(self, data_loader):
        if self.regime is not None:
            self.optimizer = adjust_optimizer(
                self.optimizer, self.epoch, self.regime)
        # switch to train mode
        self.model.train()
        output = self.feed_data(data_loader, training=True)
        return output

    def evaluate(self, data_loader):
        # switch to evaluate mode
        self.model.eval()
        return self.feed_data(data_loader, training=False)

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.epoch = checkpoint['epoch']
            self.lowet_perplexity = checkpoint['perplexity']
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         filename, self.epoch)
        else:
            logging.error('invalid checkpoint: {}'.format(filename))

    def save(self, filename='checkpoint.pth.tar', is_best=False, save_all=False):
        state = {
            'epoch': self.epoch,
            'model': self.model,
            'state_dict': self.model.state_dict(),
            'perplexity': self.lowet_perplexity,
            'regime': self.regime
        }
        state = dict(list(state.items()) + list(self.save_info.items()))
        filename = os.path.join(self.save_path, filename)
        logging.info('saving model to %s' % filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
        if save_all:
            shutil.copyfile(filename, os.path.join(
                path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))


class MultiSeq2SeqTrainer(Seq2SeqTrainer):
    """class for Trainer."""

    def iterate(self, src, target, training=True):
        src, src_length = src
        target, target_length = target
        src_full = torch.cat([src, target], 1)
        src_length_full = src_length + target_length

        target = torch.cat([target, src], 1)
        target_length_full = target_length + src_length
        src = (src_full, src_length_full)
        target = (target, target_length)
        return super(MultiSeq2SeqTrainer, self).iterate(src, target, training)

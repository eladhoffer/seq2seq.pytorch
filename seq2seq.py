import time
import logging
from itertools import chain
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from utils import *
import shutil
import pdb

from config import *


class Seq2Seq(nn.Module):
    """docstring for Trainer."""

    def __init__(self, encoder, decoder, criterion, bridge=None, optimizer=None, print_freq=10, regime=None, grad_clip=None, cuda=True):
        super(Seq2Seq, self).__init__()
        self.add_module('encoder', encoder)
        self.add_module('decoder', decoder)
        if bridge is not None:
            self.add_module('bridge', bridge)
        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=0.1)
        self.grad_clip = grad_clip
        self.epoch = 0
        self.regime = regime
        self.cuda = cuda
        self.print_freq = print_freq
        self.lowet_perplexity = None

    def encode(self, inputs, state=None):
        if state:
            return self.encoder(inputs, state)
        else:
            return self.encoder(inputs)

    def decode(self, inputs, state=None):
        return self.decoder(inputs, state)

    def forward(self, input_encoder, input_decoder):
        output, state = self.encode(input_encoder)
        if hasattr(self.modules, 'bridge'):
            input_decoder, state = self.bridge(input_decoder, state)
        return self.decode(input_decoder, state)


    def feed_data(self, data_loader, training=True):
        if training:
            assert self.optimizer is not None
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        perplexity = AverageMeter()

        end = time.time()
        for i, ((src, src_length), (target, target_length)) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if self.cuda:
                src = src.cuda()
                target = target.cuda()
            src_var = Variable(src, volatile=not training)
            target_var = Variable(target, volatile=not training)

            # compute output
            output = self(src_var, target_var[:-1])


            T, B = output.size(0), output.size(1)
            num_words = sum(target_length) - B
            loss = self.criterion(output.view(T * B, -1).contiguous(),
                                  target_var[1:].contiguous().view(-1))
            loss /= num_words
            # measure accuracy and record loss
            losses.update(loss.data[0], num_words)
            perplexity.update(2 ** loss.data[0], num_words)

            if training:
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    clip_grad_norm(self.parameters(), self.grad_clip)
                self.optimizer.step()

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

        return losses.avg, perplexity.avg

    def optimize(self, data_loader):
        if self.regime is not None:
            self.optimizer = adjust_optimizer(
                self.optimizer, self.epoch, self.regime)
        # switch to train mode
        self.train()
        output = self.feed_data(data_loader, training=True)
        self.epoch += 1
        return output

    def evaluate(self, data_loader):
        # switch to evaluate mode
        self.eval()
        return self.feed_data(data_loader, training=False)


    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['state_dict'])
            self.epoch = checkpoint['epoch']
            self.lowet_perplexity = checkpoint['perplexity']
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         filename, self.epoch)
        else:
            logging.error('invalid checkpoint: {}'.format(filename))

    def save(self, filename='checkpoint.pth.tar', path='.', is_best=False, save_all=False):
        state = {
            'epoch': self.epoch,
            'state_dict': self.state_dict(),
            'perplexity': self.lowet_perplexity,
            'regime': self.regime
        }
        filename = os.path.join(path, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
        if save_all:
            shutil.copyfile(filename, os.path.join(
                path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))

import time
import logging
from itertools import chain, cycle
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import shutil
import math
from .utils import *
from .config import PAD
from torch.nn.utils.rnn import PackedSequence

__all__ = ['Seq2SeqTrainer', 'MultiSeq2SeqTrainer', 'Img2SeqTrainer']


class AddLossModule(nn.Module):
    """adds a loss to module for easy parallelization"""

    def __init__(self, module, criterion):
        super(AddLossModule, self).__init__()
        self.module = module
        self.criterion = criterion

    def forward(self, module_inputs, target):
        output = self.module(*module_inputs)
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return self.criterion(output, target).view(1, 1)


class Seq2SeqTrainer(object):
    """class for Trainer.

     regime is an ordered list by epochs
     (can be a float indicating relative progress)"""

    def __init__(self, model, regime,
                 criterion=None,
                 print_freq=10,
                 save_freq=1000,
                 grad_clip=None,
                 embedding_grad_clip=None,
                 save_info={},
                 save_path='.',
                 checkpoint_filename='checkpoint%s.pth.tar',
                 keep_checkpoints=5,
                 devices=None,
                 cuda=True):
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss(
            size_average=False, ignore_index=PAD)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.grad_clip = grad_clip
        self.embedding_grad_clip = embedding_grad_clip
        self.epoch = 0
        self.save_info = save_info
        self.save_path = save_path
        self.save_freq = save_freq
        self.checkpoint_filename = checkpoint_filename
        self.keep_checkpoints = keep_checkpoints
        self.regime = regime
        self.current_regime_phase = None
        self.cuda = cuda
        self.print_freq = print_freq
        self.perplexity = None
        self.devices = devices
        self.model_with_loss = AddLossModule(self.model, self.criterion)
        if isinstance(self.devices, tuple):
            self.model_with_loss = DataParallel(self.model_with_loss,
                                                self.devices,
                                                dim=0 if self.batch_first else 1)

    @property
    def batch_first(self):
        return getattr(self.model.decoder, 'batch_first', False)

    def adjust_optimizer(self, epoch, total_steps):
        """Reconfigures the optimizer according to setting dict"""
        if self.regime is None:
            return
        update_optimizer = False
        if self.current_regime_phase is None:
            update_optimizer = True
            setting = {}
            # Find the first entry where the epoch is smallest than current
            for regime_phase, regime_setting in enumerate(self.regime):
                start_epoch = regime_setting.get('epoch', 0)
                start_step = regime_setting.get('step', 0)
                if epoch >= start_epoch or total_steps >= start_step:
                    self.current_regime_phase = regime_phase
                    break
        if len(self.regime) > self.current_regime_phase + 1:
            next_phase = self.current_regime_phase + 1
            # Any more regime steps?
            start_epoch = self.regime[next_phase].get('epoch', float('inf'))
            start_step = self.regime[next_phase].get('step', float('inf'))
            if epoch >= start_epoch or total_steps >= start_step:
                self.current_regime_phase = next_phase
                update_optimizer = True

        setting = self.regime[self.current_regime_phase]

        if 'lr_decay_rate' in setting and 'lr' in setting:
            decay_steps = setting.get('lr_decay_steps', 100)
            if total_steps % decay_steps == 0:
                decay_rate = setting['lr_decay_rate']
                setting['lr'] *= decay_rate ** (total_steps / decay_steps)
                update_optimizer = True
        elif 'step_lambda' in setting:
            setting = eval(setting['step_lambda'])(total_steps)
            update_optimizer = True
        elif 'epoch_lambda' in setting:
            setting = eval(setting['epoch_lambda'])(epoch)
            update_optimizer = True

        if update_optimizer:
            if 'optimizer' in setting:
                optim_method = torch.optim.__dict__[setting['optimizer']]
                if not isinstance(self.optimizer, optim_method):
                    self.optimizer = optim_method(self.optimizer.param_groups)
                    logging.debug('OPTIMIZER - setting method = %s' %
                                  setting['optimizer'])
            for param_group in self.optimizer.param_groups:
                for key in param_group.keys():
                    if key in setting:
                        new_val = setting[key]
                        if new_val != param_group[key]:
                            logging.debug('OPTIMIZER - setting %s = %s' %
                                          (key, setting[key]))
                            param_group[key] = setting[key]

    def iterate(self, src, target, training=True):
        src, src_length = src
        target, target_length = target
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        T, B = target.size(time_dim), target.size(batch_dim)
        num_words = sum(target_length) - B

        # Allow packed source sequences - for cudnn rnns
        if isinstance(src, PackedSequence):
            src_pack = src
            src = src.data
        else:
            src_pack = None

        if self.cuda and not isinstance(self.model_with_loss, DataParallel):
            src = src.cuda()
            target = target.cuda()

        src_var = Variable(src, volatile=not training)
        target_var = Variable(target, volatile=not training)

        if src_pack is not None:
            src_var = PackedSequence(src_var, src_pack[1])

        if self.batch_first:
            inputs = (src_var, target_var[:, :-1])
            target_labels = target_var[:, 1:].contiguous()
        else:
            inputs = (src_var, target_var[:-1])
            target_labels = target_var[1:]

        # compute output
        loss = self.model_with_loss(inputs, target_labels).sum()
        loss /= num_words

        if training:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                clip_grad_norm(self.model.parameters(), self.grad_clip)
            if self.embedding_grad_clip is not None and self.embedding_grad_clip > 0:
                if hasattr(self.model.encoder, 'embedder'):
                    clip_grad_norm(self.model.encoder.embedder.parameters(),
                                   self.embedding_grad_clip)
                if hasattr(self.model.decoder, 'embedder'):
                    clip_grad_norm(self.model.decoder.embedder.parameters(),
                                   self.embedding_grad_clip)
            self.optimizer.step()
        return loss.data[0], num_words

    def feed_data(self, data_loader, training=True):
        if training:
            counter = cycle(range(self.keep_checkpoints))
            total_steps = self.epoch * len(data_loader) + 1
            assert self.optimizer is not None
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        perplexity = AverageMeter()

        end = time.time()
        for i, (src, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if training:
                epoch = self.epoch + float(i) / len(data_loader)
                # update optimizer according to epoch and steps
                self.adjust_optimizer(epoch, total_steps)
                total_steps += 1

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
            if training and i % self.save_freq == 0:
                self.save_info['iteration'] = i
                self.save(identifier=next(counter))

        return losses.avg, perplexity.avg

    def optimize(self, data_loader):
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
            self.perplexity = checkpoint['perplexity']
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         filename, self.epoch)
        else:
            logging.error('invalid checkpoint: {}'.format(filename))

    def save(self, filename=None, identifier=None, is_best=False, save_all=False):
        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'perplexity': getattr(self, 'perplexity', None),
            'regime': self.regime
        }
        state = dict(list(state.items()) + list(self.save_info.items()))
        identifier = identifier or ''
        filename = filename or self.checkpoint_filename % identifier
        filename = os.path.join(self.save_path, filename)
        logging.info('saving model to %s' % filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(
                self.save_path, 'model_best.pth.tar'))
        if save_all:
            shutil.copyfile(filename, os.path.join(
                self.save_path, 'checkpoint_epoch_%s.pth.tar' % self.epoch))


class MultiSeq2SeqTrainer(Seq2SeqTrainer):
    """class for Trainer."""

    def iterate(self, src, target, training=True):
        batch_dim = 0 if self.batch_first else 1
        time_dim = 1 if self.batch_first else 0

        def pad_copy(x, length):
            if x.size(time_dim) == length:
                return x
            else:
                sz = (x.size(0), length) if self.batch_first \
                    else (length, x.size(1))
                padded = x.new().resize_(*sz).fill_(0)
                padded.narrow(time_dim, 0, x.size(time_dim)).copy_(x)
                return padded

        src, src_length = src
        target, target_length = target
        max_length = max(src.size(time_dim), target.size(time_dim))
        src = pad_copy(src, max_length)
        target = pad_copy(target, max_length)

        src_full = torch.cat([src, target], batch_dim)
        src_length_full = src_length + target_length

        target_full = torch.cat([target, src], batch_dim)
        target_length_full = target_length + src_length
        src = (src_full, src_length_full)
        target = (target_full, target_length_full)
        return super(MultiSeq2SeqTrainer, self).iterate(src, target, training)


class Img2SeqTrainer(Seq2SeqTrainer):
    """class for Trainer."""

    def iterate(self, src_img, target, training=True):
        src = (src_img, None)
        return super(Img2SeqTrainer, self).iterate(src, target, training)

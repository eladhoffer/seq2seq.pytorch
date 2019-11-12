import os
import time
import logging
from itertools import chain, cycle
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
import shutil
import math
import numpy as np
from .utils.log import ResultsLog
from .utils.optim import OptimRegime
from .utils.meters import AverageMeter
from .utils.cross_entropy import CrossEntropyLoss

from .config import PAD, UNK, EOS, BOS
from . import batch_nested_sequences
try:
    import tensorwatch
    _TENSORWATCH_AVAILABLE = True
except ImportError:
    _TENSORWATCH_AVAILABLE = False

__all__ = ['Seq2SeqTrainer', 'MultiSeq2SeqTrainer',
           'Img2SeqTrainer', 'NestedTrainer']


class AddLossModule(nn.Module):
    """adds a loss to module for easy parallelization"""

    def __init__(self, module, criterion, ignore_index=PAD):
        super(AddLossModule, self).__init__()
        self.module = module
        self.criterion = criterion
        self.ignore_index = ignore_index

    def forward(self, module_inputs, target):
        output = self.module(*module_inputs)
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        output = nn.functional.log_softmax(output, -1)
        # make sure criterion is not from_logits
        loss = self.criterion(output, target).view(1, 1)
        nll = nn.functional.nll_loss(output, target,
                                     ignore_index=self.ignore_index,
                                     reduction='sum')

        _, argmax = output.max(-1)
        invalid_targets = target.eq(self.ignore_index)
        accuracy = argmax.eq(target).masked_fill_(invalid_targets, 0)\
            .long().sum()

        return loss, nll, accuracy.view(1, 1)


def _chunk_tuple(seq_tuple, num_chunks, duplicates=1, batch_first=True):
    if num_chunks == 1:
        return [seq_tuple] * duplicates
    seq, length = seq_tuple
    batch_dim, time_dim = (0, 1) if batch_first else (1, 0)
    chunked_length = [l.tolist()
                      for l in torch.tensor(length).chunk(num_chunks)]
    chunks = []
    for chunk, chunk_length in zip(seq.chunk(num_chunks, dim=batch_dim), chunked_length):
        # chunk = chunk.narrow(time_dim, 0, max(chunk_length))
        chunks.append((chunk, chunk_length))
    return chunks * duplicates


def _batch_max_tokens(src_tuple, target_tuple, max_tokens, batch_first=True, log=True):
    src, src_length = src_tuple
    target, target_length = target_tuple
    num_tokens = src.size(0) * src.size(1) + \
        target.size(0) * target.size(1)
    if num_tokens > max_tokens:
        batch_dim, time_dim = (0, 1) if batch_first else (1, 0)
        batch_size = src.size(batch_dim)
        src_max_length = np.maximum.accumulate(src_length)
        target_max_length = np.maximum.accumulate(target_length)
        sum_max_length = src_max_length + target_max_length
        num_tokens_batch = sum_max_length * \
            (np.arange(src.size(batch_dim)) + 1)
        B = int((num_tokens_batch > max_tokens).argmax() - 1)
        if B < 0:
            B = batch_size
        Tsrc = int(src_max_length[B - 1])
        Ttarget = int(target_max_length[B - 1])
        src = src.narrow(batch_dim, 0, B).narrow(time_dim, 0, Tsrc)
        src_tuple = (src, src_length[:B])

        target = target.narrow(batch_dim, 0, B).narrow(time_dim, 0, Ttarget)
        target_tuple = (target, target_length[:B])
        if log and B < batch_size:
            logging.debug(
                'Trimmed batch to %s as number of tokens was > %s, T = (%s, %s)' %
                (B, max_tokens, Tsrc, Ttarget))
    return src_tuple, target_tuple


class DecodedInputTargets(object):
    def __init__(self, batch_first=True):
        self.batch_first = batch_first
        self.time_dim = 1 if batch_first else 0
        self.training = True

    def train(self, enable=True):
        self.training = enable

    def eval(self):
        self.training = False

    def __call__(self, src_pair, target_pair):
        src, src_len = src_pair
        target, target_len = target_pair
        return src, target, target


class TeacherForcing(DecodedInputTargets):

    def __call__(self, src_pair, target_pair):
        src, inputs, targets = super(
            TeacherForcing, self).__call__(src_pair, target_pair)
        if self.batch_first:
            inputs = inputs[:, :-1]
            targets = targets[:, 1:].contiguous()
        else:
            inputs = inputs[:-1]
            targets = targets[1:]
        return src, inputs, targets


class Seq2SeqTrainer(object):
    """class for Trainer.

     regime is an ordered list by epochs
     (can be a float indicating relative progress)"""

    def __init__(self, model, regime=None,
                 criterion=None,
                 label_smoothing=0,
                 target_forcing='teacher',
                 print_freq=10,
                 eval_freq=1000,
                 save_freq=600,
                 save_every_epoch=False,
                 grad_clip=None,
                 embedding_grad_clip=None,
                 max_tokens=None,
                 chunk_batch=1,
                 duplicates=1,
                 save_info={},
                 save_path='.',
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 avg_loss_time=True,
                 distributed=False,
                 local_rank=0,
                 dtype=torch.float,
                 loss_scale=1,
                 device_ids=None,
                 device="cuda"):
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion or CrossEntropyLoss(
            ignore_index=PAD, smooth_eps=label_smoothing, reduction='sum', from_logits=False)

        self.optimizer = OptimRegime(
            self.model, regime=regime, use_float_copy=dtype == torch.float16)

        if target_forcing == 'teacher':
            self.target_forcing = TeacherForcing(batch_first=self.batch_first)
        else:
            self.target_forcing = DecodedInputTargets()
        self.grad_clip = grad_clip
        self.embedding_grad_clip = embedding_grad_clip
        self.epoch = 0
        self.training_steps = 0
        self.save_info = save_info
        self.device = device
        self.dtype = dtype
        self.loss_scale = loss_scale
        self.max_tokens = max_tokens
        self.chunk_batch = chunk_batch
        self.duplicates = duplicates
        self.print_freq = print_freq
        self.eval_freq = eval_freq
        self.save_every_epoch = save_every_epoch
        self.perplexity = float('inf')
        self.device_ids = device_ids
        self.avg_loss_time = avg_loss_time
        self.model_with_loss = AddLossModule(self.model, self.criterion)
        self.distributed = distributed
        self.local_rank = local_rank
        if distributed:
            self.model_with_loss = DistributedDataParallel(
                self.model_with_loss,
                device_ids=[local_rank],
                find_unused_parameters=False,
                output_device=local_rank)
        else:
            if isinstance(self.device_ids, tuple):
                self.model_with_loss = DataParallel(self.model_with_loss,
                                                    self.device_ids,
                                                    dim=0 if self.batch_first else 1)
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_counter = cycle(range(keep_checkpoints))
        self.last_save = time.time()
        self.checkpoint_filename = checkpoint_filename
        results_file = os.path.join(save_path, 'results')
        self.results = ResultsLog(results_file,
                                  params=save_info.get('config', None))
        self.watcher = None
        self.streams = {}

    @property
    def batch_first(self):
        return getattr(self.model.decoder, 'batch_first', False)

    def iterate(self, src_tuple_batch, target_tuple_batch, training=True, chunk_batch=1):
        loss_measure = 0
        accuracy_measure = 0
        nll_measure = 0
        num_words = 0
        batch_dim = 0 if self.batch_first else 1
        if training:
            self.optimizer.zero_grad()
            self.target_forcing.train()
        else:
            self.target_forcing.eval()

        repacked_inputs = []
        for src_tuple, target_tuple in zip(_chunk_tuple(src_tuple_batch, chunk_batch, self.duplicates, self.batch_first),
                                           _chunk_tuple(target_tuple_batch, chunk_batch, self.duplicates, self.batch_first)):
            # limit number of tokens to avoid gpu overload
            if training and self.max_tokens is not None:
                src_tuple, target_tuple = _batch_max_tokens(
                    src_tuple, target_tuple, self.max_tokens,
                    batch_first=self.batch_first)
            encoded, decoded, target = self.target_forcing(
                src_tuple, target_tuple)
            num_words += int(target.ne(PAD).sum())
            repacked_inputs.append((encoded, decoded, target))

        for encoded, decoded, target in repacked_inputs:
            if not isinstance(self.model_with_loss, DataParallel):
                encoded = encoded.to(self.device)
                target = target.to(self.device)
                decoded = decoded.to(self.device)

            if training:
                self.optimizer.pre_forward()
            # compute output
            inputs = (encoded, decoded)
            loss, nll, accuracy = self.model_with_loss(inputs, target)
            loss = loss.sum()

            if self.avg_loss_time:
                loss /= num_words
            else:
                loss /= (target.size(batch_dim) * len(repacked_inputs))

            if training:
                self.optimizer.pre_backward()
                if self.loss_scale > 1:
                    loss *= self.loss_scale
                # compute gradient and do SGD step
                loss.backward()

                if self.loss_scale > 1:
                    for p in self.model.parameters():
                        p.grad.data.div_(self.loss_scale)

            accuracy_measure += float(accuracy.sum().float() / num_words)
            nll_measure += float(nll.sum() / num_words)
            loss_measure += float(loss)

        if training:
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, dict):
                    clip_encoder = self.grad_clip.get('encoder', 0)
                    clip_decoder = self.grad_clip.get('decoder', 0)
                    if clip_encoder > 0:
                        clip_grad_norm_(
                            self.model.encoder.parameters(), clip_encoder)
                    if clip_decoder > 0:
                        clip_grad_norm_(
                            self.model.decoder.parameters(), clip_decoder)
                elif self.grad_clip > 0:  # grad_clip is a number
                    clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
            if self.embedding_grad_clip is not None and self.embedding_grad_clip > 0:
                if hasattr(self.model.encoder, 'embedder'):
                    clip_grad_norm_(self.model.encoder.embedder.parameters(),
                                    self.embedding_grad_clip)
                if hasattr(self.model.decoder, 'embedder'):
                    clip_grad_norm_(self.model.decoder.embedder.parameters(),
                                    self.embedding_grad_clip)
            self.optimizer.step()
        return loss_measure, nll_measure, accuracy_measure, num_words

    def _feed_data(self, data_loader, num_iterations=None, training=True, chunk_batch=1):
        if training:
            assert self.optimizer is not None

        num_iterations = num_iterations or len(data_loader) - 1
        meters = {
            'batch': AverageMeter(),
            'data': AverageMeter(),
            'tokens': AverageMeter(),
            'loss': AverageMeter(),
            'perplexity': AverageMeter(),
            'accuracy': AverageMeter()
        }

        end = time.time()
        for i, (src, target) in enumerate(data_loader):
            # measure data loading time
            meters['data'].update(time.time() - end)

            try:
                if training:
                    self.epoch += 1. / len(data_loader)
                    if self.distributed:
                        data_loader.sampler.set_epoch(
                            int(math.floor(self.epoch)))
                    self.training_steps += 1
                    # update optimizer according to epoch and steps
                    self.optimizer.update(self.epoch, self.training_steps)
                # do a train/evaluate iteration
                loss, nll, acc, num_words = self.iterate(src, target,
                                                         training=training,
                                                         chunk_batch=chunk_batch)

                # measure accuracy and record loss
                meters['loss'].update(loss, num_words)
                meters['perplexity'].update(math.exp(nll), num_words)
                meters['accuracy'].update(acc, num_words)

                # measure elapsed time
                elapsed = time.time() - end
                meters['batch'].update(elapsed)
                meters['tokens'].update(num_words / elapsed)

                end = time.time()
            except RuntimeError as err:
                if training and 'out of memory' in str(err):
                    logging.info(
                        'WARNING: ran out of memory, skipping batch')
                    torch.cuda.empty_cache()
                else:
                    raise err

            last_iteration = (i == len(data_loader) - 1)

            if training:
                if (end - self.last_save) > self.save_freq:
                    self.last_save = end
                    self.save(identifier=next(self.save_counter))
                if last_iteration and self.save_every_epoch:
                    self.save(identifier='epoch_%s' % int(self.epoch))

            if i > 0 or last_iteration:
                if hasattr(self.target_forcing, 'keep_prob'):
                    prob = self.target_forcing.keep_prob
                    self.target_forcing.keep_prob = max(0., prob - 1e-5)

                if i % self.print_freq == 0 or last_iteration:
                    if hasattr(self.target_forcing, 'keep_prob'):
                        logging.info('keep-prob = %s' %
                                     self.target_forcing.keep_prob)
                    logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                                 'Time {meters[batch].val:.3f} ({meters[batch].avg:.3f})\t'
                                 'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                                 'Tok/sec {meters[tokens].val:.3f} ({meters[tokens].avg:.3f})\t'
                                 'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                                 'Accuracy {meters[accuracy].val:.4f} ({meters[accuracy].avg:.4f})\t'
                                 'Perplexity {meters[perplexity].val:.4f} ({meters[perplexity].avg:.4f})'.format(
                                     int(self.epoch), i, len(data_loader),
                                     phase='TRAINING' if training else 'EVALUATING',
                                     meters=meters))
                    self.observe(trainer=self,
                                 model=self.model,
                                 optimizer=self.optimizer,
                                 data=(src, target))
                    self.stream_meters(meters,
                                       prefix='train' if training else 'eval')
                    if hasattr(self.model.decoder, 'output_order'):
                        logging.info(self.model.decoder.output_order[0])
                        logging.info(self.model.decoder.output_order[1])
                        logging.info(self.model.decoder.output_order[0][0])
                        logging.info(self.model.decoder.output_order[1][0])
                        logging.info(self.model.decoder.output_order[2][0])
                    if training:
                        self.write_stream('lr',
                                          (self.training_steps, self.optimizer.get_lr()[0]))

                if i % num_iterations == 0 or last_iteration:
                    yield dict([(name, meter.avg) for name, meter in meters.items()])
                    for meter in meters.values():
                        meter.reset()

    def optimize(self, data_loader):
        # switch to train mode
        self.model.train()
        for result in self._feed_data(
                data_loader,
                num_iterations=self.eval_freq,
                chunk_batch=self.chunk_batch,
                training=True):
            yield result
            self.model.train()

    def evaluate(self, data_loader):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            for r in self._feed_data(data_loader, training=False):
                result = r
        return result

    def run(self, train_loader, val_loader=None):
        for train_results in self.optimize(train_loader):
            results = {'epoch': self.epoch,
                       'training steps': self.training_steps,
                       'training loss': train_results['loss'],
                       'training accuracy': train_results['accuracy'],
                       'training perplexity': train_results['perplexity']}
            plot_loss = ['training loss']
            plot_perplexity = ['training perplexity']
            plot_accuracy = ['training accuracy']
            if val_loader is not None:
                # evaluate on validation set
                val_results = self.evaluate(val_loader)

                # remember best prec@1 and save checkpoint
                is_best = val_results['perplexity'] < self.perplexity
                if is_best:
                    self.perplexity = val_results['perplexity']
                    self.save(is_best=True)

                results['validation loss'] = val_results['loss']
                results['validation perplexity'] = val_results['perplexity']
                results['validation accuracy'] = val_results['accuracy']
                plot_loss += ['validation loss']
                plot_perplexity += ['validation perplexity']
                plot_accuracy += ['validation accuracy']

            if self.distributed and torch.distributed.get_rank() > 0:
                continue
            self.results.add(**results)
            self.results.plot(x='training steps', y=plot_perplexity,
                              title='Perplexity', ylabel='perplexity')
            self.results.plot(x='training steps', y=plot_accuracy,
                              title='Accuracy', ylabel='accuracy')
            self.results.plot(x='training steps', y=plot_loss,
                              title='Loss', ylabel='loss')
            self.results.plot(x='epoch', y=plot_perplexity,
                              title='Perplexity', ylabel='perplexity')
            self.results.plot(x='epoch', y=plot_accuracy,
                              title='Accuracy', ylabel='accuracy')
            self.results.plot(x='epoch', y=plot_loss,
                              title='Loss', ylabel='loss')
            self.results.save()

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.epoch = checkpoint['epoch']
            self.training_steps = checkpoint['training_steps']
            self.perplexity = checkpoint['perplexity']
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         filename, self.epoch)
        else:
            logging.error('invalid checkpoint: {}'.format(filename))

    def save(self, filename=None, identifier=None, is_best=False, save_all=False):
        if self.distributed and torch.distributed.get_rank() > 0:  # avoid multiple writes
            return
        state = {
            'epoch': self.epoch,
            'training_steps': self.training_steps,
            'state_dict': self.model.state_dict(),
            'perplexity': getattr(self, 'perplexity', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))
        identifier = identifier or ''
        filename = filename or self.checkpoint_filename % identifier
        filename = os.path.join(self.save_path, filename)
        logging.info('saving model to %s' % filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(
                self.save_path, 'model_best.pth'))
        if save_all:
            shutil.copyfile(filename, os.path.join(
                self.save_path, 'checkpoint_epoch_%s.pth' % self.epoch))

    ###### tensorwatch methods to enable training-time logging ######

    def set_watcher(self, filename, port=0):
        if not _TENSORWATCH_AVAILABLE:
            return False
        if self.distributed and self.local_rank > 0:
            return False
        self.watcher = tensorwatch.Watcher(filename=filename, port=port)
        # default streams
        self._default_streams()
        self.watcher.make_notebook()
        return True

    def get_stream(self, name, **kwargs):
        if self.watcher is None:
            return None
        if name not in self.streams.keys():
            self.streams[name] = self.watcher.create_stream(name=name,
                                                            **kwargs)
        return self.streams[name]

    def write_stream(self, name, values):
        stream = self.get_stream(name)
        if stream is not None:
            stream.write(values)

    def stream_meters(self, meters_dict, prefix=None):
        if self.watcher is None:
            return False
        for name, value in meters_dict.items():
            if prefix is not None:
                name = '_'.join([prefix, name])
            value = value.val
            stream = self.get_stream(name)
            if stream is None:
                continue
            stream.write((self.training_steps, value))
        return True

    def observe(self, **kwargs):
        if self.watcher is None:
            return False
        self.watcher.observe(**kwargs)
        return True

    def _default_streams(self):
        self.get_stream('train_loss')
        self.get_stream('eval_loss')
        self.get_stream('train_accuracy')
        self.get_stream('eval_accuracy')
        self.get_stream('train_perplexity')
        self.get_stream('eval_perplexity')
        self.get_stream('lr')


class MultiSeq2SeqTrainer(Seq2SeqTrainer):
    """class for Trainer."""

    def iterate(self, src, target, training=True, chunk_batch=1):
        assert chunk_batch == 1
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

    def iterate(self, src_img, target, training=True, chunk_batch=1):
        src = (src_img, None)
        return super(Img2SeqTrainer, self).iterate(src, target, training)


class NestedTrainer(Seq2SeqTrainer):
    """class for Trainer.

     regime is an ordered list by epochs
     (can be a float indicating relative progress)"""

    def __init__(self, *kargs, **kwargs):
        super(NestedTrainer, self).__init__(*kargs, **kwargs)
        self.model_with_loss = AddLossModule(self.model, self.criterion)
        if self.distributed:
            self.model_with_loss = DistributedDataParallel(
                self.model_with_loss,
                device_ids=[self.local_rank],
                output_device=self.local_rank)
        else:
            if isinstance(self.device_ids, tuple):
                self.model_with_loss = DataParallel(self.model_with_loss,
                                                    self.device_ids,
                                                    dim=0 if self.batch_first else 1)
        _, target_tok = self.save_info['tokenizers'].values()
        target_words = target_tok.common_words(8188)
        self.contrast_batch = batch_nested_sequences(target_words)

    def iterate(self, src_tuple, target_tuple, training=True):
        # limit number of tokens to avoid gpu overload
        if self.max_tokens is not None:
            src_tuple, target_tuple = self._batch_max_tokens(
                src_tuple, target_tuple)
        (src_word, src_word_length), (src_char, src_char_length) = src_tuple
        (target_word, target_word_length), (target_char, target_char_length) \
            = target_tuple
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        num_words = sum(target_word_length) - target.size(batch_dim)

        src = src_char
        target = target_word

        if not isinstance(self.model_with_loss, DataParallel):
            src = src.to(self.device)
            target = target.to(self.device)

        if self.batch_first:
            inputs = (src, target[:, :-1])
            target_labels = target[:, 1:].contiguous()
        else:
            inputs = (src, target[:-1])
            target_labels = target[1:]

        # compute output
        loss, nll, accuracy = self.model_with_loss(inputs, target_labels)

        loss = loss.sum()
        loss_measure = float(loss / num_words)
        if self.avg_loss_time:
            loss /= num_words
        else:
            loss /= target.size(batch_dim)
        accuracy = float(accuracy.sum().float() / num_words)

        if training:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, dict):
                    clip_encoder = self.grad_clip.get('encoder', 0)
                    clip_decoder = self.grad_clip.get('decoder', 0)
                    if clip_encoder > 0:
                        clip_grad_norm_(
                            self.model.encoder.parameters(), clip_encoder)
                    if clip_decoder > 0:
                        clip_grad_norm_(
                            self.model.decoder.parameters(), clip_decoder)
                elif self.grad_clip > 0:  # grad_clip is a number
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if self.embedding_grad_clip is not None and self.embedding_grad_clip > 0:
                if hasattr(self.model.encoder, 'embedder'):
                    clip_grad_norm_(self.model.encoder.embedder.parameters(),
                                    self.embedding_grad_clip)
                if hasattr(self.model.decoder, 'embedder'):
                    clip_grad_norm_(self.model.decoder.embedder.parameters(),
                                    self.embedding_grad_clip)
            self.optimizer.step()
        return loss_measure, accuracy, num_words

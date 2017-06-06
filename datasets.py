# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from copy import copy, deepcopy
import math
import logging
from tokenizer import Tokenizer, BPETokenizer, CharTokenizer
from config import *
import torch
from data import LinedTextDataset
from collections import OrderedDict

__tokenizers = {
    'word': Tokenizer,
    'char': CharTokenizer,
    'bpe': BPETokenizer
}


def create_padded_batch(max_length=100):
    def collate(seqs):
        if not torch.is_tensor(seqs[0]):
            return tuple([collate(s) for s in zip(*seqs)])
        lengths = [min(len(s), max_length) for s in seqs]
        batch_length = max(lengths)
        seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(PAD)
        for i, s in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])
        return (seq_tensor, lengths)
    return collate


def create_sorted_batches(max_length=100):
    def collate(seqs):
        seqs.sort(key=lambda p: len(p), reverse=True)
        lengths = [min(len(s), max_length) for s in seqs]
        batch_length = max(lengths)
        seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(PAD)
        for i, s in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])

        return (seq_tensor, lengths)
    return collate


class MultiLanguageDataset(object):
    """docstring for Dataset."""

    def __init__(self, prefix="./data/OpenSubtitles2016.en-he",
                 languages=['en', 'he'],
                 tokenization='bpe',
                 num_symbols=32000,
                 shared_vocab=True,
                 code_files=None,
                 vocab_files=None,
                 insert_start=[BOS], insert_end=[EOS],
                 tokenizers=None,
                 load_data=True):
        super(MultiLanguageDataset, self).__init__()
        self.languages = languages
        self.shared_vocab = shared_vocab
        self.num_symbols = num_symbols
        self.tokenizers = tokenizers
        self.tokenization = tokenization
        self.insert_start = insert_start
        self.insert_end = insert_end
        self.input_files = {l: '{prefix}.{lang}'.format(
            prefix=prefix, lang=l) for l in languages}

        if self.tokenizers is None:
            if tokenization not in ['bpe', 'char', 'word']:
                raise ValueError("An invalid option for tokenization was used, options are {0}".format(
                    ','.join(['bpe', 'char', 'word'])))


            if tokenization == 'bpe':
                if not shared_vocab:
                    self.code_files = code_files or {l: '{prefix}.{lang}.{tok}.codes_{num_symbols}'.format(
                        prefix=prefix, lang=l, tok=tokenization, num_symbols=num_symbols) for l in languages}
                else:
                    code_file = code_files or '{prefix}.{tok}.shared_codes_{num_symbols}_{languages}'.format(
                        prefix=prefix, tok=tokenization, languages='_'.join(languages), num_symbols=num_symbols)
                    self.code_files = {l: code_file for l in languages}

            if not shared_vocab:
                self.vocab_files = vocab_files or {l: '{prefix}.{lang}.{tok}.vocab{num_symbols}'.format(
                    prefix=prefix, lang=l, tok=tokenization, num_symbols=num_symbols) for l in languages}
            else:
                vocab = vocab_files or '{prefix}.{tok}.shared_vocab{num_symbols}_{languages}'.format(
                    prefix=prefix, tok=tokenization, languages='_'.join(languages), num_symbols=num_symbols)
                self.vocab_files = {l: vocab for l in languages}
            self.generate_tokenizers()

        if load_data:
            self.load_data()

    def generate_tokenizers(self):
        self.tokenizers = OrderedDict()
        for l in self.languages:
            if self.shared_vocab:
                files = [self.input_files[t] for t in self.languages]
            else:
                files = self.input_files[l]

            if self.tokenization == 'bpe':
                tokz = BPETokenizer(self.code_files[l],
                                    vocab_file=self.vocab_files[l],
                                    num_symbols=self.num_symbols)
                if not hasattr(tokz, 'bpe'):
                    tokz.learn_bpe(files)
            else:
                tokz = __tokenizers[self.tokenization](
                    vocab_file=self.vocab_files[l])

            if not hasattr(tokz, 'vocab'):
                logging.info('generating vocabulary. saving to %s' %
                             self.vocab_files[l])
                tokz.get_vocab(files)
                tokz.save_vocab(self.vocab_files[l])
            self.tokenizers[l] = tokz

    def load_data(self):
        self.datasets = OrderedDict()
        for l in self.languages:
            transform = lambda t: self.tokenizers[l].tokenize(
                t, insert_start=self.insert_start, insert_end=self.insert_end)
            self.datasets[l] = LinedTextDataset(
                self.input_files[l], transform=transform)

    def split(self, ratio=0.5):
        data_size = len(self)
        num_remove = 30000  # int(float(data_size) * ratio)
        num_keep = data_size - num_remove
        data1 = copy(self)
        data2 = copy(self)
        for d in self.languages:
            data1.datasets[d] = self.datasets[d].narrow(0, num_keep)
            data2.datasets[d] = self.datasets[d].narrow(num_keep, num_remove)
        return data1, data2

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        items = []
        for d in self.languages:
            items.append(self.datasets[d][index])
        return tuple(items)

    def __len__(self):
        return len(self.datasets[self.languages[0]])

    def get_loader(self, languages=None, batch_size=1, shuffle=False, sampler=None, num_workers=0,
                   collate_fn=create_padded_batch(), pin_memory=False, drop_last=False):
        languages = languages or self.languages
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)


class WMT16_de_en(MultiLanguageDataset):
    """docstring for Dataset."""

    def __init__(self, root='./data/wmt16_de_en',
                 split='train',
                 tokenization='bpe',
                 num_symbols=32000,
                 shared_vocab=True,
                 code_files=None,
                 vocab_files=None,
                 insert_start=[BOS],
                 insert_end=[EOS],
                 tokenizers=None,
                 load_data=True):

        train_prefix = "{root}/train.clean".format(root=root)
        options = dict(
                     prefix=train_prefix,
                     languages=['de', 'en'],
                     tokenization=tokenization,
                     num_symbols=num_symbols,
                     shared_vocab=shared_vocab,
                     code_files=code_files,
                     vocab_files=vocab_files,
                     insert_start=insert_start,
                     insert_end=insert_end,
                     tokenizers=tokenizers,
                     load_data=False
                     )
        train_options = deepcopy(options)

        if split == 'train':
            options = train_options
        else:
            train_data = MultiLanguageDataset(**train_options)
            options['tokenizers'] = getattr(train_data, 'tokenizers', None)
            options['code_files'] = getattr(train_data, 'code_files', None)
            options['vocab_files'] = getattr(train_data, 'vocab_files', None)
            if split == 'dev':
                prefix="{root}/newstest2014.clean".format(root=root)
            elif split == 'test':
                prefix="{root}/newstest2016.clean".format(root=root)

            options['prefix'] = prefix
        super(WMT16_de_en, self).__init__(**options)
        if load_data:
            self.load_data()

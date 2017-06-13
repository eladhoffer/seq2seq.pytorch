# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import logging
import sys
import os
from copy import copy, deepcopy
import torch
from collections import OrderedDict
from .text import LinedTextDataset
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from tools.tokenizer import Tokenizer, BPETokenizer, CharTokenizer
from tools.config import *

DATA_PATH = './data'
__tokenizers = {
    'word': Tokenizer,
    'char': CharTokenizer,
    'bpe': BPETokenizer
}


def create_padded_batch(max_length=100, batch_first=False, sort=False, pack=False):
    def collate(seqs):
        if not torch.is_tensor(seqs[0]):
            return tuple([collate(s) for s in zip(*seqs)])
        if sort or pack:  # packing requires a sorted batch by length
            seqs.sort(key=lambda p: len(p), reverse=True)
        lengths = [min(len(s), max_length) for s in seqs]
        batch_length = max(lengths)
        seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(PAD)
        for i, s in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])
        if batch_first:
            seq_tensor = seq_tensor.t()
        if pack:
            return pack_padded_sequence(seq_tensor, lengths, batch_first=batch_first)
        else:
            return (seq_tensor, lengths)
    return collate


class MultiLanguageDataset(object):
    """docstring for Dataset."""

    def __init__(self, prefix,
                 languages,
                 tokenization='bpe',
                 num_symbols=32000,
                 shared_vocab=True,
                 code_files=None,
                 vocab_files=None,
                 insert_start=[BOS], insert_end=[EOS],
                 mark_language=False,
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
        self.mark_language = mark_language
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
                        prefix=prefix, tok=tokenization, languages='_'.join(sorted(languages)), num_symbols=num_symbols)
                    self.code_files = {l: code_file for l in languages}

            if not shared_vocab:
                self.vocab_files = vocab_files or {l: '{prefix}.{lang}.{tok}.vocab{num_symbols}'.format(
                    prefix=prefix, lang=l, tok=tokenization, num_symbols=num_symbols) for l in languages}
            else:
                vocab = vocab_files or '{prefix}.{tok}.shared_vocab{num_symbols}_{languages}'.format(
                    prefix=prefix, tok=tokenization, languages='_'.join(sorted(languages)), num_symbols=num_symbols)
                self.vocab_files = {l: vocab for l in languages}
            self.generate_tokenizers()

        if load_data:
            self.load_data()

    def generate_tokenizers(self):
        self.tokenizers = OrderedDict()
        additional_tokens = None
        if self.mark_language:
            additional_tokens = [LANGUAGE_TOKENS[l] for l in self.languages]
        for l in self.languages:
            if self.shared_vocab:
                files = [self.input_files[t] for t in self.languages]
            else:
                files = self.input_files[l]

            if self.tokenization == 'bpe':
                tokz = BPETokenizer(self.code_files[l],
                                    vocab_file=self.vocab_files[l],
                                    num_symbols=self.num_symbols,
                                    additional_tokens=additional_tokens)
                if not hasattr(tokz, 'bpe'):
                    tokz.learn_bpe(files)
            else:
                tokz = __tokenizers[self.tokenization](
                    vocab_file=self.vocab_files[l],
                    additional_tokens=additional_tokens)

            if not hasattr(tokz, 'vocab'):
                logging.info('generating vocabulary. saving to %s' %
                             self.vocab_files[l])
                tokz.get_vocab(files)
                tokz.save_vocab(self.vocab_files[l])
            self.tokenizers[l] = tokz

    def load_data(self):
        self.datasets = OrderedDict()
        for l in self.languages:
            insert_start = deepcopy(self.insert_start)
            if self.mark_language:
                lang_idx = self.tokenizers[l]\
                    .special_tokens.index(LANGUAGE_TOKENS[l])
                insert_start.append(lang_idx)
            insert_end = self.insert_end

            def transform(t, insert_start=insert_start, insert_end=insert_end):
                return self.tokenizers[l].tokenize(t,
                                                   insert_start=insert_start,
                                                   insert_end=insert_end)
            self.datasets[l] = LinedTextDataset(
                self.input_files[l], transform=transform)

    def select_range(self, start, end):
        new_dataset = copy(self)
        new_dataset.datasets = dict(
            {l: d.select_range(start, end) for (l, d) in self.datasets.items()})
        return new_dataset

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
                   max_length=100, batch_first=False, pin_memory=False, drop_last=False):
        collate_fn = create_padded_batch(
            max_length=max_length, batch_first=batch_first)
        languages = languages or self.languages
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)

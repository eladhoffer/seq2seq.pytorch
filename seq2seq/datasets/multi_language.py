# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import logging
from copy import copy, deepcopy
import torch
from collections import OrderedDict
from .text import LinedTextDataset
from seq2seq.tools.tokenizer import Tokenizer, BPETokenizer, CharTokenizer, SentencePiece, WordCharTokenizer
from seq2seq.tools.config import *
from seq2seq.tools import batch_sequences, batch_nested_sequences


def create_padded_batch(max_length=100, max_tokens=None, fixed_length=None,
                        batch_first=False, sort=False,
                        pack=False, augment=False):
    def collate(seqs, sort=sort, pack=pack):
        if not torch.is_tensor(seqs[0]):
            if sort or pack:  # packing requires a sorted batch by length
                # sort by the first set
                seqs.sort(key=lambda x: len(x[0]), reverse=True)
            # TODO: for now, just the first input will be packed
            return tuple([collate(s, sort=False, pack=pack and (i == 0))
                          for i, s in enumerate(zip(*seqs))])
        return batch_sequences(seqs, max_length=max_length,
                               max_tokens=max_tokens,
                               fixed_length=fixed_length,
                               batch_first=batch_first,
                               sort=False, pack=pack, augment=augment)
    return collate


def create_nested_padded_batch(max_length=100, max_tokens=None, fixed_length=None,
                               batch_first=True, sort=False,
                               pack=False, augment=False):
    assert batch_first and not any([sort, pack, augment])

    def collate(seqs, first=True):
        return tuple([batch_nested_sequences(lng, max_length=max_length,
                                             max_tokens=max_tokens,
                                             fixed_length=fixed_length,
                                             batch_first=batch_first) for lng in zip(*seqs)])
    return collate


class MultiLanguageDataset(object):
    """docstring for Dataset."""
    __tokenizers = {
        'word': Tokenizer,
        'char': CharTokenizer,
        'bpe': BPETokenizer,
        'sentencepiece': SentencePiece,
        'word+char': WordCharTokenizer,
    }

    def __init__(self, prefix,
                 languages,
                 tokenization='bpe',
                 tokenization_config={},
                 tokenization_model_files=None,
                 shared_vocab=True,
                 insert_start=[BOS], insert_end=[EOS],
                 mark_language=False,
                 tokenizers=None,
                 vocab_limit=None,
                 load_data=True,
                 sample=False):
        super(MultiLanguageDataset, self).__init__()
        if tokenization.startswith('moses+'):
            tokenization = tokenization.replace('moses+', '')
            mark_moses_pretok = True
        self.languages = languages
        self.shared_vocab = shared_vocab
        self.tokenizers = tokenizers
        self.tokenization = tokenization
        self.tokenization_config = tokenization_config
        self.insert_start = insert_start
        self.insert_end = insert_end
        self.vocab_limit = vocab_limit
        self.mark_language = mark_language
        self.sample = sample
        self.input_files = {l: '{prefix}.{lang}'.format(
            prefix=prefix, lang=l) for l in languages}

        if tokenization == 'bpe' or tokenization == 'sentencepiece':
            self.tokenization_config.setdefault('num_symbols', 32000)

        tok_setting = [tokenization]
        if mark_moses_pretok:
            tok_setting = ['moses_pretok'] + tok_setting
        for n, v in tokenization_config.items():
            if (isinstance(v, bool) and v):
                tok_setting.append(n)
            elif n == 'model_type' or n == 'num_symbols':
                tok_setting.append(str(v))
            else:
                tok_setting.append('%s-%s' % (n, v))

        tok_prefix = '.'.join(tok_setting)

        if self.tokenizers is None:
            if tokenization not in ['sentencepiece', 'bpe', 'char', 'word', 'word+char']:
                raise ValueError("An invalid option for tokenization was used, options are {0}".format(
                    ','.join(['sentencepiece', 'bpe', 'char', 'word'])))

            if not shared_vocab:
                self.tok_files = tokenization_model_files or {l: '{prefix}.{tok}.{lang}'.format(
                    prefix=prefix, lang=l, tok=tok_prefix) for l in languages}
            else:
                tok_file = tokenization_model_files or '{prefix}.{tok}.shared-{languages}'.format(
                    prefix=prefix, tok=tok_prefix, languages='_'.join(sorted(languages)))
                self.tok_files = {l: tok_file for l in languages}

            if self.mark_language:
                self.tokenization_config['additional_tokens'] = \
                    [LANGUAGE_TOKENS(l) for l in self.languages]
            self.generate_tokenizers()

        if load_data:
            self.load_data()

    def generate_tokenizers(self):
        self.tokenizers = OrderedDict()

        for l in self.languages:
            if self.shared_vocab:
                files = [self.input_files[t] for t in self.languages]
            else:
                files = [self.input_files[l]]
            tok_config = deepcopy(self.tokenization_config)
            tok_config['file_prefix'] = self.tok_files[l]
            tokz = self.__tokenizers[self.tokenization](**tok_config)
            if self.tokenization == 'sentencepiece':
                if getattr(tokz, 'model', None) is None:
                    tokz.learn_model(files)
            else:
                if self.tokenization == 'bpe' and not hasattr(tokz, 'bpe'):
                    tokz.learn_bpe(files)

                if getattr(tokz, 'vocab', None) is None:
                    logging.info('generating vocabulary. saving to %s' %
                                 self.tok_files[l])
                    tokz.get_vocab(files)
                    tokz.save_vocab(self.tok_files[l])
                tokz.load_vocab(self.tok_files[l], limit=self.vocab_limit)
            self.tokenizers[l] = tokz

    def load_data(self):
        self.datasets = OrderedDict()
        for l in self.languages:
            insert_start = self.insert_start
            if self.mark_language:
                lang_idx = self.tokenizers[l].special_tokens.index(
                    LANGUAGE_TOKENS(l))
                insert_start = [lang_idx]
            insert_end = self.insert_end

            def transform(txt, tokenizer=self.tokenizers[l],
                          insert_start=insert_start,
                          insert_end=insert_end,
                          sample=self.sample):
                return tokenizer.tokenize(txt,
                                          insert_start=insert_start,
                                          insert_end=insert_end,
                                          sample=sample)
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

    def get_loader(self, batch_size=1, shuffle=False, sort=False, pack=False,
                   augment=False, languages=None, sampler=None, num_workers=0,
                   max_length=100, max_tokens=None, fixed_length=None, batch_first=False,
                   pin_memory=False, drop_last=False):
        if self.tokenization == 'word+char':
            collate_fn = create_nested_padded_batch(
                max_length=max_length, max_tokens=max_tokens, batch_first=batch_first,
                fixed_length=fixed_length, sort=sort, pack=pack, augment=augment)
        else:
            collate_fn = create_padded_batch(
                max_length=max_length, max_tokens=max_tokens, batch_first=batch_first,
                fixed_length=fixed_length, sort=sort, pack=pack, augment=augment)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)

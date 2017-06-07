# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import string
import codecs
import logging
import sys
from collections import Counter
import torch
from .config import *
sys.path.append(".datasets/subword-nmt")
import learn_bpe
import apply_bpe


class Tokenizer(object):

    def __init__(self, max_length=500, vocab_file=None, vocab_threshold=2):
        self.max_length = max_length
        self.vocab_threshold = vocab_threshold
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.__word2idx = {}
        if os.path.isfile(vocab_file):
            self.load_vocab(vocab_file)

    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    def idx2word(self, idx):
        if idx < len(self.special_tokens):
            return self.special_tokens[idx]
        else:
            return self.vocab[idx - len(self.special_tokens)][0]

    def update_word2idx(self):
        self.__word2idx = {
            word[0]: idx + len(self.special_tokens) for idx, word in enumerate(self.vocab)}
        for i, tok in enumerate(self.special_tokens):
            self.__word2idx[tok] = i

    def word2idx(self, word):
        return self.__word2idx.get(word, UNK)

    def segment(self, line):
        """segments a line to tokenizable items"""
        return str(line).lower().translate(string.punctuation).strip().split()

    def get_vocab(self, filenames, limit=None):
        # get combined vocabulary of all input texts
        vocab = Counter()
        for fname in filenames:
            with codecs.open(fname, encoding='UTF-8') as f:
                for line in f:
                    for word in self.segment(line):
                        vocab[word] += 1
        self.vocab = vocab.most_common(limit)
        self.update_word2idx()

    def save_vocab(self, vocab_filename):
        if self.vocab is not None:
            with codecs.open(vocab_filename, 'w', encoding='UTF-8') as f:
                for (key, freq) in self.vocab:
                    f.write("{0} {1}\n".format(key, freq))

    def load_vocab(self, vocab_filename, limit=None):
        vocab = Counter()
        with codecs.open(vocab_filename, encoding='UTF-8') as f:
            for line in f:
                word, count = line.strip().split()
                vocab[word] = int(count)
        self.vocab = vocab.most_common(limit)
        self.update_word2idx()

    def tokenize(self, line, insert_start=None, insert_end=None):
        """tokenize a line, insert_start and insert_end are lists of tokens"""
        inputs = self.segment(line)
        targets = []
        if insert_start is not None:
            targets += insert_start
        for w in inputs:
            targets.append(self.word2idx(w))
        if insert_end is not None:
            targets += insert_end
        return torch.LongTensor(targets)

    def detokenize(self, inputs, delimeter=' '):
        return delimeter.join([self.idx2word(idx) for idx in inputs])


class BPETokenizer(Tokenizer):

    def __init__(self, codes_file, vocab_file,
                 num_symbols=10000, min_frequency=2, seperator='@@'):
        super(BPETokenizer, self).__init__(vocab_file=vocab_file)
        self.num_symbols = num_symbols
        self.min_frequency = min_frequency
        self.seperator = seperator
        self.codes_file = codes_file
        if os.path.isfile(codes_file):
            self.set_bpe(codes_file)

    def set_bpe(self, codes_file):
        with codecs.open(self.codes_file, encoding='UTF-8') as codes:
            self.bpe = apply_bpe.BPE(codes, self.seperator, None)

    def segment(self, line):
        if not hasattr(self, 'bpe'):
            raise NameError('Learn bpe first!')
        return self.bpe.segment(line).strip().split()

    def learn_bpe(self, filenames):
        logging.info('generating bpe codes file. saving to %s' % self.codes_file)
        if isinstance(filenames, str):
            filenames = [filenames]

        # get combined vocabulary of all input texts
        full_vocab = Counter()
        for fname in filenames:
            with codecs.open(fname, encoding='UTF-8') as f:
                full_vocab += learn_bpe.get_vocabulary(f)
        vocab_list = ['{0} {1}'.format(key, freq)
                      for (key, freq) in full_vocab.items()]
        # pdb.set_trace()
        # learn BPE on combined vocabulary
        with codecs.open(self.codes_file, 'w', encoding='UTF-8') as output:
            learn_bpe.main(vocab_list, output, self.num_symbols,
                           self.min_frequency, False, is_dict=True)
        self.set_bpe(self.codes_file)

    def detokenize(self, inputs, delimeter=' '):
        detok_string = super(BPETokenizer, self).detokenize(inputs, delimeter)
        return detok_string.replace(self.seperator + ' ', '')


class CharTokenizer(Tokenizer):

    def __init__(self, vocab_file):
        super(CharTokenizer, self).__init__(vocab_file=vocab_file)

    def segment(self, line):
        return list(line.strip())

    def detokenize(self, inputs, delimeter=''):
        return super(CharTokenizer, self).detokenize(inputs, delimeter)

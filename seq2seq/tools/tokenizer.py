# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import string
import codecs
import logging
import sys
from collections import Counter, OrderedDict
import torch
from .config import *

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), './subword-nmt')))
import learn_bpe
import apply_bpe


class OrderedCounter(Counter, OrderedDict):
    pass


class Tokenizer(object):

    def __init__(self, max_length=500, vocab_file=None,
                 additional_tokens=None, pre_tokenize=None, post_tokenize=None,
                 vocab_threshold=2):
        self.max_length = max_length
        self.vocab_threshold = vocab_threshold
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.pre_tokenize = pre_tokenize
        self.post_tokenize = post_tokenize

        if additional_tokens is not None:
            self.special_tokens += additional_tokens
        self.__word2idx = {}
        if os.path.isfile(vocab_file):
            self.load_vocab(vocab_file)
        # if use_moses is not None
        # try:
        #     from nltk.tokenize.moses import MosesDetokenizer
        # except ImportError:
        #     raise ImportError('nltk is needed for moses tokenization')

        # tknzr = MosesTokenizer(lang)

    @property
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
        line = str(line)
        return line.strip().split()

    def get_vocab(self, item_list, from_filenames=True, limit=None):
        vocab = OrderedCounter()
        if from_filenames:
            filenames = item_list
            # get combined vocabulary of all input files
            for fname in filenames:
                with codecs.open(fname, encoding='UTF-8') as f:
                    for line in f:
                        for word in self.segment(line):
                            vocab[word] += 1
        else:
            for line in item_list:
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
        vocab = OrderedCounter()
        with codecs.open(vocab_filename, encoding='UTF-8') as f:
            for line in f:
                try:
                    word, count = line.strip().split()
                except:  # no count
                    word, count = line.strip(), 0
                vocab[word] = int(count)
        self.vocab = vocab.most_common(limit)
        self.update_word2idx()

    def tokenize(self, line, insert_start=None, insert_end=None):
        """tokenize a line, insert_start and insert_end are lists of tokens"""
        if getattr(self, 'pre_tokenize', None) is not None:
            line = self.pre_tokenize(line)
        inputs = self.segment(line)
        targets = []
        if insert_start is not None:
            targets += insert_start
        for w in inputs:
            targets.append(self.word2idx(w))
        if insert_end is not None:
            targets += insert_end
        return torch.LongTensor(targets)

    def detokenize(self, inputs, delimiter=u' '):
        outputs = delimiter.join([self.idx2word(idx) for idx in inputs])
        if getattr(self, 'post_tokenize', None) is not None:
            outputs = self.post_tokenize(outputs)
        return outputs


class BPETokenizer(Tokenizer):

    def __init__(self, codes_file, vocab_file, additional_tokens=None,
                 num_symbols=10000, min_frequency=2, separator='@@',
                 pre_tokenize=None, post_tokenize=None,):
        super(BPETokenizer, self).__init__(vocab_file=vocab_file,
                                           additional_tokens=additional_tokens,
                                           pre_tokenize=pre_tokenize,
                                           post_tokenize=post_tokenize)
        self.num_symbols = num_symbols
        self.min_frequency = min_frequency
        self.separator = separator
        self.codes_file = codes_file
        if os.path.isfile(codes_file):
            self.set_bpe(codes_file)

    def set_bpe(self, codes_file):
        with codecs.open(self.codes_file, encoding='UTF-8') as codes:
            self.bpe = apply_bpe.BPE(codes, separator=self.separator)

    def segment(self, line):
        if not hasattr(self, 'bpe'):
            raise NameError('Learn bpe first!')
        return self.bpe.segment(line.strip()).split()

    def learn_bpe(self, item_list, from_filenames=True):
        logging.info('generating bpe codes file. saving to %s' %
                     self.codes_file)
        if from_filenames:
            filenames = item_list
            if isinstance(filenames, str):
                filenames = [filenames]

            # get combined vocabulary of all input files
            full_vocab = OrderedCounter()
            for fname in filenames:
                with codecs.open(fname, encoding='UTF-8') as f:
                    full_vocab += learn_bpe.get_vocabulary(f)
        else:
            # get combined vocabulary of all input texts
            full_vocab = OrderedCounter()
            full_vocab += learn_bpe.get_vocabulary(item_list)

        vocab_list = ['{0} {1}'.format(key, freq)
                      for (key, freq) in full_vocab.items()]
        # learn BPE on combined vocabulary
        with codecs.open(self.codes_file, 'w', encoding='UTF-8') as output:
            learn_bpe.main(vocab_list, output, self.num_symbols,
                           self.min_frequency, False, is_dict=True)
        self.set_bpe(self.codes_file)

    def detokenize(self, inputs, delimiter=' '):
        self.separator = getattr(self, 'separator', '@@')
        detok_string = super(BPETokenizer, self).detokenize(inputs, delimiter)
        try:
            detok_string = detok_string.decode('utf-8')
        except:
            pass
        detok_string = detok_string\
            .replace(self.separator + ' ', '')\
            .replace(self.separator, '')
        return detok_string


class CharTokenizer(Tokenizer):

    def segment(self, line):
        return list(line.strip())

    def detokenize(self, inputs, delimiter=u''):
        return super(CharTokenizer, self).detokenize(inputs, delimiter)

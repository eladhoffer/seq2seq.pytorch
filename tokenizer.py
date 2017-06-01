from __future__ import unicode_literals

import os
import torch
import string
import codecs
import sys
from collections import Counter
import pdb
sys.path.append("./subword-nmt")
import learn_bpe, apply_bpe

class Tokenizer(object):
    def __init__(self, max_length=500, vocab_file=None, vocab_threshold=2):
        self.UNK_TOKEN = '<unk>'
        self.PAD_TOKEN = '<pad>'
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '<\s>'
        self.UNK, self.PAD, self.BOS, self.EOS = [0, 1, 2, 3]
        self.max_length = 500
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        if os.path.isfile(vocab_file):
            self.load_vocab(vocab_file)

    def idx2word(self, idx):
        if idx < len(self.special_tokens):
            return self.special_tokens[idx]
        else:
            return self.vocab[idx+len(self.special_tokens)]

    def word2idx(self, word):
        if not hasattr(self, '__word2idx'):
            self.__word2idx = {word[0]: idx+len(self.special_tokens) for idx, word in enumerate(self.vocab)}
            for i, tok in enumerate(self.special_tokens):
                self.__word2idx[tok] = i
        return self.__word2idx.get(w, self.UNK)


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

    def tokenize(self, line):
        """tokenize a line"""
        inputs = self.segment(line)
        for w in inputs:
            targets.append(self.word2idx(w))
        return torch.LongTensor(targets)

    def detokenize(self, inputs, delimeter=' '):
        return delimeter.join([self.idx2word(idx) for idx in inputs])

class BPETokentizer(Tokenizer):
    def __init__(self, codes_file, vocab_file,
                 num_symbols=10000, min_frequency=2, seperator='@@'):
        super(BPETokentizer, self).__init__(vocab_file=vocab_file)
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
        if isinstance(filenames, str):
            filenames = [filenames]

        # get combined vocabulary of all input texts
        full_vocab = Counter()
        for fname in filenames:
            with codecs.open(fname, encoding='UTF-8') as f:
                full_vocab += learn_bpe.get_vocabulary(f)
        vocab_list = ['{0} {1}'.format(key, freq) for (key, freq) in full_vocab.items()]
        # pdb.set_trace()
        # learn BPE on combined vocabulary
        with codecs.open(self.codes_file, 'w', encoding='UTF-8') as output:
            learn_bpe.main(vocab_list, output, self.num_symbols, self.min_frequency, False, is_dict=True)
        self.set_bpe(self.codes_file)


if __name__ == '__main__':
    #
    # # python 2/3 compatibility
    # if sys.version_info < (3, 0):
    #     sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    #     sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    #     sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    # else:
    #     sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    #     sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    #     sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    prefix = "/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en"
    langs = ['en', 'fr']
    num_symbols = 32000
    shared_vocab = True
    input_files = ['{prefix}.{lang}'.format(prefix=prefix,lang=l) for l in langs]
    if not shared_vocab:
        code_files =  ['{prefix}.{lang}.codes_{num_symbols}'.format(prefix=prefix,lang=l,num_symbols=num_symbols) for l in langs]
        vocabs = ['{prefix}.{lang}.vocab{num_symbols}'.format(prefix=prefix,lang=l,num_symbols=num_symbols) for l in langs]
    else:
        code_files = '{prefix}.shared_codes_{num_symbols}_{langs}'.format(prefix=prefix,langs='_'.join(langs),num_symbols=num_symbols)
        code_files = [code_files] * len(langs)
        vocabs = '{prefix}.shared_vocab{num_symbols}_{langs}'.format(prefix=prefix,langs='_'.join(langs),num_symbols=num_symbols)
        vocabs = [vocabs] * len(langs)


    tokenizers = []
    for i,t in enumerate(langs):
        tokz = BPETokentizer(code_files[i], vocab_file=vocabs[i], num_symbols=num_symbols)
        if shared_vocab:
            files = input_files
        else:
            files = input_files[i]
        if not hasattr(tokz, 'bpe'):
            tokz.learn_bpe(files)
        if not hasattr(tokz, 'vocab'):
            tokz.get_vocab(files)
            tokz.save_vocab(vocabs[i])
        tokenizers.append(tokz)

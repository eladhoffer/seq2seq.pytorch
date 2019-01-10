# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import string
import codecs
import logging
import tempfile
import sys
from collections import Counter, OrderedDict
import torch
from .config import *

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), './subword-nmt')))
import learn_bpe
import apply_bpe

try:
    from sacremoses import MosesTokenizer, MosesDetokenizer
    _MOSES_AVAILABLE = True
except ImportError:
    _MOSES_AVAILABLE = False

try:
    import sentencepiece as spm
    _SENTENCEPIECE_AVAILABLE = True
except ImportError:
    _SENTENCEPIECE_AVAILABLE = False


class OrderedCounter(Counter, OrderedDict):
    pass


def _segment_words(line, pre_apply=None):
    if pre_apply is not None:
        line = pre_apply(line)
    line = str(line)
    return line.strip('\r\n ').split()


def _get_vocabulary(item_list, segment=_segment_words, from_filenames=True):
    vocab = OrderedCounter()
    if from_filenames:
        filenames = item_list
        # get combined vocabulary of all input files
        for fname in filenames:
            with codecs.open(fname, encoding='UTF-8') as f:
                for line in f:
                    for word in segment(line):
                        vocab[word] += 1
    else:
        for line in item_list:
            for word in segment(line):
                vocab[word] += 1
    return vocab


class Tokenizer(object):

    def __init__(self, vocab_file=None, additional_tokens=None, use_moses=None):
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        if use_moses is not None:
            self.enable_moses(lang=use_moses)
        if additional_tokens is not None:
            self.special_tokens += additional_tokens
        self.__word2idx = {}
        self.vocab_file = vocab_file
        if os.path.isfile(vocab_file):
            self.load_vocab(vocab_file)

    def enable_moses(self, lang='en', tokenize=True, detokenize=True):
        if tokenize:
            self._moses_tok = MosesTokenizer(lang=lang)
        else:
            self._moses_tok = None

        if detokenize:
            self._moses_detok = MosesDetokenizer(lang=lang)
        else:
            self._moses_detok = None

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    def pre_tokenize(self, line):
        if hasattr(self, '_moses_tok'):
            return self._moses_tok.tokenize(line, return_str=True)
        return line

    def post_detokenize(self, tokens):
        if hasattr(self, '_moses_detok'):
            return self._moses_detok.detokenize(tokens, return_str=False)
        return tokens

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

    def segment(self, line, sample=None):
        """segments a line to tokenizable items"""
        line = self.pre_tokenize(line)
        return _segment_words(line)

    def get_vocab(self, item_list, from_filenames=True, limit=None):
        vocab = _get_vocabulary(item_list=item_list, segment=self.segment,
                                from_filenames=from_filenames)
        self.vocab = vocab.most_common(limit)
        self.update_word2idx()

    def save_vocab(self, vocab_filename):
        if self.vocab is not None:
            with codecs.open(vocab_filename, 'w', encoding='UTF-8') as f:
                for (key, freq) in self.vocab:
                    f.write("{0} {1}\n".format(key, freq))

    def load_vocab(self, vocab_filename, limit=None, min_count=1):
        vocab = OrderedCounter()
        with codecs.open(vocab_filename, encoding='UTF-8') as f:
            for line in f:
                try:
                    word, count = line.strip().split()
                except:  # no count
                    word, count = line.strip(), 1
                count = int(count)
                if count >= min_count:
                    vocab[word] = count
        self.vocab = vocab.most_common(limit)
        self.update_word2idx()

    def tokenize(self, line, insert_start=None, insert_end=None, sample=None):
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

    def detokenize(self, inputs, delimiter=u' '):
        token_list = [self.idx2word(int(idx)) for idx in inputs]
        token_list = self.post_detokenize(token_list)
        outputs = delimiter.join(token_list)
        return outputs


class BPETokenizer(Tokenizer):

    def __init__(self, codes_file, vocab_file, additional_tokens=None,
                 num_symbols=10000, min_frequency=2, total_symbols=False, separator='@@',
                 **kwargs):
        super(BPETokenizer, self).__init__(vocab_file=vocab_file,
                                           additional_tokens=additional_tokens,
                                           **kwargs)
        self.num_symbols = num_symbols
        self.min_frequency = min_frequency
        self.total_symbols = total_symbols
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
        line = self.pre_tokenize(line)
        return self.bpe.segment(line.strip('\r\n ')).split(' ')

    def learn_bpe(self, item_list, from_filenames=True):
        logging.info('generating bpe codes file. saving to %s' %
                     self.codes_file)

        # get vocabulary at word level (before bpe)
        def segment_words(line): return _segment_words(line, self.pre_tokenize)
        vocab_words = _get_vocabulary(item_list,
                                      from_filenames=from_filenames,
                                      segment=segment_words)
        vocab_list = ['{0} {1}'.format(key, freq)
                      for (key, freq) in vocab_words.items()]
        # learn BPE on combined vocabulary
        with codecs.open(self.codes_file, 'w', encoding='UTF-8') as output:
            learn_bpe.learn_bpe(vocab_list, output, num_symbols=self.num_symbols,
                                min_frequency=self.min_frequency, verbose=False,
                                is_dict=True, total_symbols=self.total_symbols)
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
        line = self.pre_tokenize(line)
        return list(line.strip())

    def detokenize(self, inputs, delimiter=u''):
        return super(CharTokenizer, self).detokenize(inputs, delimiter)


class SentencePiece(Tokenizer):
    def __init__(self, model_prefix, additional_tokens=None,
                 num_symbols=10000, model_type='unigram',
                 character_coverage=None, split_by_whitespace=False):
        assert _SENTENCEPIECE_AVAILABLE
        self.model_prefix = os.path.abspath(model_prefix)
        self.num_symbols = num_symbols
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.split_by_whitespace = split_by_whitespace
        self.vocab_file = '{}.vocab'.format(model_prefix)
        self.model_file = '{}.model'.format(model_prefix)
        self.model = None
        if additional_tokens is not None:
            self.special_tokens += additional_tokens
        if os.path.isfile(self.model_file):
            self.load_model(self.model_file)

    def serialize_model(self, model_file):
        with open(model_file, 'rb') as f:
            return f.read()

    def deserialize_model(self, model_serialized):
        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(model_serialized)
            self.load_model(path)
        finally:
            os.remove(path)

    def load_model(self, model_file):
        self.model = spm.SentencePieceProcessor()
        self.model.Load(model_file)
        self._model_serialized = self.serialize_model(model_file)

    def learn_model(self, file_list, **kwargs):
        file_list = ','.join([os.path.abspath(filename)
                              for filename in file_list])
        self.train_sp_model(input=file_list,
                            model_prefix=self.model_prefix,
                            vocab_size=self.num_symbols,
                            character_coverage=self.character_coverage,
                            model_type=self.model_type,
                            split_by_whitespace=self.split_by_whitespace,
                            **kwargs)
        self.load_model(self.model_file)

    def tokenize(self, line, insert_start=None, insert_end=None, sample=None):
        """tokenize a line, insert_start and insert_end are lists of tokens"""
        if sample is None or sample is False:
            targets = self.model.EncodeAsIds(line)
        else:
            sample = sample if isinstance(sample, dict) else {}
            sample.setdefault('nbest_size', 64)
            sample.setdefault('alpha', 0.1)
            targets = self.model.SampleEncodeAsIds(line, **sample)
        if insert_start is not None:
            targets = insert_start + targets
        if insert_end is not None:
            targets += insert_end
        return torch.LongTensor(targets)

    def detokenize(self, inputs):
        outputs = self.model.DecodeIds([int(idx) for idx in inputs])
        return outputs

    def idx2word(self, idx):
        return self.model.IdToPiece(idx)

    def word2idx(self, word):
        return self.model.PieceToId(word)

    def segment(self, line, sample=None):
        """segments a line to tokenizable items"""
        if sample is None or sample is False:
            return self.model.EncodeAsPieces(line)
        else:
            sample = sample if isinstance(sample, dict) else {}
            sample.setdefault('nbest_size', 64)
            sample.setdefault('alpha', 0.1)
            return self.model.SampleEncodeAsPieces(line, **sample)

    @property
    def vocab_size(self):
        return len(self.model)

    @staticmethod
    def train_sp_model(**kwargs):
        """possible arguments:
        --input: one-sentence-per-line raw corpus file. You can pass a comma-separated list of files.
        --model_prefix: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
        --vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
        --character_coverage: amount of characters covered by the model
        --model_type: model type. Choose from unigram (default), bpe, char, or word. The input sentence must be pretokenized when using word type.
        """
        kwargs.update({'unk_piece': UNK_TOKEN, 'bos_piece': BOS_TOKEN,
                       'eos_piece': EOS_TOKEN, 'pad_piece': PAD_TOKEN,
                       'unk_id': UNK, 'bos_id': BOS,
                       'eos_id': EOS, 'pad_id': PAD
                       })
        for arg, val in kwargs.items():
            if isinstance(val, bool):
                kwargs[arg] = 'true' if val else 'false'
        config = ' '.join(['--{}={}'.format(name, value)
                           for name, value in kwargs.items() if value is not None])
        spm.SentencePieceTrainer.Train(config)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def __setstate__(self, newstate):
        self.deserialize_model(newstate['_model_serialized'])


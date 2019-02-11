# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
import os
from seq2seq.tools.config import *
from copy import deepcopy
from .multi_language import MultiLanguageDataset


class OpenSubtitles2018(MultiLanguageDataset):
    """docstring for Dataset."""

    def __init__(self,
                 root,
                 languages,
                 split='train',
                 tokenization='bpe',
                 use_moses=False,
                 num_symbols=32000,
                 shared_vocab=True,
                 code_files=None,
                 vocab_files=None,
                 insert_start=[BOS],
                 insert_end=[EOS],
                 mark_language=False,
                 tokenizers=None,
                 load_data=True):

        prefix = os.path.join(root, '-'.join(sorted(languages)) + '/{}.clean')
        train_prefix = prefix.format('train')
        options = dict(
            prefix=train_prefix,
            languages=languages,
            tokenization=tokenization,
            use_moses=use_moses,
            num_symbols=num_symbols,
            shared_vocab=shared_vocab,
            code_files=code_files,
            vocab_files=vocab_files,
            insert_start=insert_start,
            insert_end=insert_end,
            mark_language=mark_language,
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
            options['prefix'] = prefix.format(split)

        super(OpenSubtitles2018, self).__init__(**options)
        if load_data:
            self.load_data()

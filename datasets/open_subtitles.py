# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from tools.config import *
from copy import deepcopy
from .multi_language import MultiLanguageDataset


class OpenSubtitles2016(MultiLanguageDataset):
    """docstring for Dataset."""

    def __init__(self,
                 root,
                 languages,
                 tokenization='bpe',
                 num_symbols=32000,
                 shared_vocab=True,
                 code_files=None,
                 vocab_files=None,
                 insert_start=[BOS],
                 insert_end=[EOS],
                 tokenizers=None,
                 load_data=True):

        options = dict(
            prefix=root + '.' + '-'.join(sorted(languages)),
            languages=languages,
            tokenization=tokenization,
            num_symbols=num_symbols,
            shared_vocab=shared_vocab,
            code_files=code_files,
            vocab_files=vocab_files,
            insert_start=insert_start,
            insert_end=insert_end,
            tokenizers=tokenizers,
            load_data=load_data
        )

        super(OpenSubtitles2016, self).__init__(**options)

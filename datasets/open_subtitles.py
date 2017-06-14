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
                 split='train',
                 tokenization='bpe',
                 num_symbols=32000,
                 shared_vocab=True,
                 code_files=None,
                 vocab_files=None,
                 insert_start=[BOS],
                 insert_end=[EOS],
                 mark_language=False,
                 tokenizers=None,
                 load_data=True,
                 dev_size=3000,
                 test_size=3000):

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
            mark_language=mark_language,
            tokenizers=tokenizers,
            load_data=load_data
        )

        super(OpenSubtitles2016, self).__init__(**options)
        if split == 'train':
            self = self.select_range(0, len(self) - (dev_size + test_size + 1))
        elif split == 'dev':
            self = self.select_range(
                len(self) - (dev_size + test_size), len(self) - (test_size - 1))
        elif split == 'test':
            self = self.select_range(len(self) - test_size, len(self))

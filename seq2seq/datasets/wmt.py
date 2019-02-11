# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from seq2seq.tools.config import *
from copy import deepcopy
from .multi_language import MultiLanguageDataset


class WMT(MultiLanguageDataset):
    """WMT dataset class"""

    def __init__(self,
                 root,
                 split='train',
                 tokenization='bpe',
                 tokenization_config={},
                 tokenization_model_files=None,
                 shared_vocab=True,
                 insert_start=[BOS],
                 insert_end=[EOS],
                 mark_language=False,
                 tokenizers=None,
                 vocab_limit=None,
                 moses_pretok=True,
                 languages=['en', 'de'],
                 train_file="{root}/train{pretok}.clean",
                 val_file="{root}/newstest2014{pretok}.clean",
                 test_file="{root}/newstest2016{pretok}.clean",
                 load_data=True,
                 sample=False):
        pretok = ''
        sample = sample if split == 'train' else False
        if moses_pretok:
            pretok = '.tok'
            tokenization = 'moses+' + tokenization
        train_prefix = train_file.format(root=root, pretok=pretok)
        options = dict(
            prefix=train_prefix,
            languages=languages,
            tokenization=tokenization,
            tokenization_config=tokenization_config,
            tokenization_model_files=tokenization_model_files,
            shared_vocab=shared_vocab,
            insert_start=insert_start,
            insert_end=insert_end,
            mark_language=mark_language,
            tokenizers=tokenizers,
            vocab_limit=vocab_limit,
            load_data=False,
            sample=sample
        )
        train_options = deepcopy(options)

        if split == 'train':
            options = train_options
        else:
            train_data = MultiLanguageDataset(**train_options)
            options['tokenizers'] = getattr(train_data, 'tokenizers', None)
            options['tokenization_model_files'] = getattr(
                train_data, 'tokenization_model_files', None)
            if split == 'dev':
                prefix = val_file.format(root=root, pretok=pretok)
            elif split == 'test':
                prefix = test_file.format(root=root, pretok=pretok)

            options['prefix'] = prefix
        super(WMT, self).__init__(**options)
        if load_data:
            self.load_data()


class WMT16_de_en(WMT):
    """docstring for Dataset."""

    def __init__(self, *kargs, **kwargs):
        super(WMT16_de_en, self).__init__(*kargs, **kwargs)


class WMT17_de_en(WMT):
    """docstring for Dataset."""

    def __init__(self, *kargs, **kwargs):
        kwargs.setdefault('test_file', "{root}/newstest2017{pretok}.clean")
        super(WMT17_de_en, self).__init__(*kargs, **kwargs)

# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import string
import codecs
import sys
import torch
from datasets import MultiLanguageDataset, WMT16_de_en, OpenSubtitles2016
from torch.utils.data import DataLoader

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

    # prefix = './data/wmt16_de_en/train.clean'
    # langs = ['en', 'de']
    #
    # dataset = MultiLanguageDataset(prefix=prefix, languages=langs)
    # train_data = WMT16_de_en(split='train')
    # dev_data = WMT16_de_en(split='dev')
    data = OpenSubtitles2016(
        root='/home/ehoffer/PyTorch/seq2seq.pytorch/datasets/data/OpenSubtitles2016', languages=['en', 'he'])


# check_single_batch = DataLoader(datasets[i], batch_size=32, collate_fn=create_padded_batch())
# check = AlignedDatasets(datasets)

from __future__ import unicode_literals

import os
import string
import codecs
import sys
import torch
from datasets import MultiLanguageDataset
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

    # prefix = "/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en"
    langs = ['en', 'he']

    dataset = MultiLanguageDataset(languages=langs)

# check_single_batch = DataLoader(datasets[i], batch_size=32, collate_fn=create_padded_batch())
# check = AlignedDatasets(datasets)

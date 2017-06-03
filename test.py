from __future__ import unicode_literals

import os
import string
import codecs
import sys
import torch
from data import LinedTextDataset, AlignedDatasets, create_padded_batch, get_dataset_bpe
from tokenizer import BPETokentizer
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
    # langs = ['en', 'fr']

datasets, tokenizers = get_dataset_bpe(append_bos=[True, False])
check_batch = DataLoader(AlignedDatasets(
    datasets), batch_size=32, collate_fn=create_padded_batch(), shuffle=True, num_workers=8)

# check_single_batch = DataLoader(datasets[i], batch_size=32, collate_fn=create_padded_batch())
# check = AlignedDatasets(datasets)

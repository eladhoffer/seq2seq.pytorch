import os
from copy import copy
import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import string
from random import randrange
from config import *
import codecs
import sys
from tokenizer import BPETokenizer


def list_line_locations(filename):
    line_offset = []
    offset = 0
    with open(filename, "rb") as f:
        for line in f:
            line_offset.append(offset)
            offset += len(line)
    return line_offset


class LinedTextDataset(Dataset):

    def __init__(self, filename, transform=None, load_mem=False):
        self.filename = filename
        self.load_mem = load_mem
        self.transform = transform
        if self.load_mem:
            self.items = []
            with codecs.open(self.filename, encoding='UTF-8') as f:
                for line in f:
                    self.items.append(line)
        else:
            self.items = list_line_locations(filename)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        if self.load_mem:
            item = self.items[index]
        else:
            with codecs.open(self.filename, encoding='UTF-8') as f:
                f.seek(self.items[index])
                item = f.readline()
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.items)

    def narrow(self, start, num):
        new_dataset = copy(self)
        new_dataset.items = new_dataset.items[start:(start + num - 1)]
        return new_dataset

#
# class AlignedDatasets(Dataset):
#
#     def __init__(self, datasets):
#         self.datasets = datasets
#
#     def __getitem__(self, index):
#         items = []
#         for dataset in self.datasets:
#             items.append(dataset[index])
#         return tuple(items)
#
#     def __len__(self):
#         return len(self.datasets[0])
#
#
# class NarrowDataset(Dataset):
#
#     def __init__(self, dataset, first_item=0, last_item=None):
#         self.dataset = dataset
#         last_item = last_item or len(self.dataset) - 1
#         self.first_item = min(max(first_item, 0), len(self.dataset) - 1)
#         self.last_item = min(max(last_item, 0), len(self.dataset) - 1)
#
#     def __getitem__(self, index):
#         return self.dataset[index + self.first_item]
#
#     def __len__(self):
#         return self.last_item - self.first_item + 1

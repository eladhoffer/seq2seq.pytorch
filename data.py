import os
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

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


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
            with open(self.filename, encoding="utf-8") as f:
                for line in f:
                    self.items.append(line)
        else:
            self.items = list_line_locations(filename)

    def __getitem__(self, index):
        if self.load_mem:
            item = self.items[index]
        else:
            with open(self.filename, encoding="utf-8") as f:
                f.seek(self.items[index])
                item = f.readline()
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.items)


class AlignedDatasets(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        items = []
        for dataset in self.datasets:
            items.append(dataset[index])
        return tuple(items)

    def __len__(self):
        return len(self.datasets[0])

class NarrowDataset(Dataset):

    def __init__(self, dataset, first_item=0, last_item=None):
        self.dataset = dataset
        last_item = last_item or len(self.dataset) - 1
        self.first_item = min(max(first_item, 0), len(self.dataset)-1)
        self.last_item = min(max(last_item, 0), len(self.dataset)-1)

    def __getitem__(self, index):
        return self.dataset[index + self.first_item]

    def __len__(self):
        return self.last_item - self.first_item + 1


def create_padded_batch(max_length=100):
    def collate(seqs):
        if not torch.is_tensor(seqs[0]):
            return tuple([collate(s) for s in zip(*seqs)])
        lengths = [min(len(s), max_length) for s in seqs]
        batch_length = max(lengths)
        seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(PAD)
        for i, s in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])
        return (seq_tensor, lengths)
    return collate

def create_sorted_batches(max_length=100):
    def collate(seqs):
        seqs.sort(key=lambda p: len(p), reverse=True)
        lengths = [min(len(s), max_length) for s in seqs]
        batch_length = max(lengths)
        seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(PAD)
        for i, s in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])

        return (seq_tensor, lengths)
    return collate



def get_dataset_bpe(prefix="./data/OpenSubtitles2016.en-he",
                    langs=['en', 'he'],
                    num_symbols=32000,
                    shared_vocab=True,
                    append_bos=None, append_eos=None):
    append_bos = append_bos or [True] * len(langs)
    append_eos = append_eos or [True] * len(langs)
    input_files = ['{prefix}.{lang}'.format(
        prefix=prefix, lang=l) for l in langs]
    if not shared_vocab:
        code_files = ['{prefix}.{lang}.codes_{num_symbols}'.format(
            prefix=prefix, lang=l, num_symbols=num_symbols) for l in langs]
        vocabs = ['{prefix}.{lang}.vocab{num_symbols}'.format(
            prefix=prefix, lang=l, num_symbols=num_symbols) for l in langs]
    else:
        code_files = '{prefix}.shared_codes_{num_symbols}_{langs}'.format(
            prefix=prefix, langs='_'.join(langs), num_symbols=num_symbols)
        code_files = [code_files] * len(langs)
        vocabs = '{prefix}.shared_vocab{num_symbols}_{langs}'.format(
            prefix=prefix, langs='_'.join(langs), num_symbols=num_symbols)
        vocabs = [vocabs] * len(langs)

    tokenizers = []
    for i, t in enumerate(langs):
        tokz = BPETokenizer(code_files[i], vocab_file=vocabs[
                             i], num_symbols=num_symbols)
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

    datasets = []
    for i in range(len(input_files)):
        transform = lambda t: tokenizers[i].tokenize(
            t, append_bos=append_bos[i], append_eos=append_eos[i])
        datasets.append(LinedTextDataset(
            input_files[i], transform=transform))
    return datasets, tokenizers

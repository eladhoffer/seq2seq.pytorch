from copy import copy
import codecs
import torch
from torch.utils.data import Dataset
from seq2seq.tools import batch_sequences


def list_line_locations(filename):
    line_offset = []
    offset = 0
    with open(filename, "rb") as f:
        for line in f:
            line_offset.append(offset)
            offset += len(line)
    return line_offset


def list_word_locations(filename):
    word_offset = []
    offset = 0
    with open(filename, "rb") as f:
        for line in f:
            for i, _ in enumerate(line.split()):
                word_offset.append((offset, i))
            offset += len(line)
    return word_offset

class LinedTextDataset(Dataset):
    """ Dataset in which every line is a seperate item (e.g translation)
    """

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

    def select_range(self, start, end):
        new_dataset = copy(self)
        new_dataset.items = new_dataset.items[start:end]
        return new_dataset

    def filter(self, filter_func):
        new_dataset = copy(self)
        new_dataset.items = [item for item in self if filter_func(item)]
        return new_dataset

    def get_loader(self, sort=False, pack=False,
                   batch_size=1, shuffle=False, sampler=None, num_workers=0,
                   max_length=None, batch_first=False, pin_memory=False, drop_last=False):
        def collate_fn(seqs): return batch_sequences(seqs, max_length=max_length,
                                                     batch_first=batch_first,
                                                     sort=sort, pack=pack)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)


class WordedTextDataset(LinedTextDataset):
    """ Dataset in which every line is a seperate item (e.g translation)
    """

    def __init__(self, filename, transform=None):
        self.filename = filename
        self.load_mem = False
        self.transform = transform

        self.items = list_word_locations(filename)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        with codecs.open(self.filename, encoding='UTF-8') as f:
            f.seek(self.items[index][0])
            item = f.readline()
            item = list(item.split())[self.items[index][1]]
        if self.transform is not None:
            item = self.transform(item)
        return item

class TextFileDataset(Dataset):
    """ Dataset in which data is a continuous chunk of text
    """

    def __init__(self, filename, transform=lambda x: x.split()):
        self.filename = filename
        self.transform = transform
        self.items = []
        with codecs.open(self.filename, encoding='UTF-8') as f:
            for line in f:
                self.items += transform(line)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def select_range(self, start, end):
        new_dataset = copy(self)
        new_dataset.items = new_dataset.items[start:end]
        return new_dataset

    def filter(self, filter_func):
        new_dataset = copy(self)
        new_dataset.items = [item for item in self if filter_func(item)]
        return new_dataset

    def get_loader(self, sort=False, pack=False,
                   batch_size=1, shuffle=False, sampler=None, num_workers=0,
                   max_length=None, batch_first=False, pin_memory=False, drop_last=False):
        def collate_fn(seqs): return batch_sequences(seqs, max_length=max_length,
                                                     batch_first=batch_first,
                                                     sort=sort, pack=pack)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)

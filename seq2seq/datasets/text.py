from copy import copy
from torch.utils.data import Dataset
import codecs
import sys


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

    def select_range(self, start, end):
        new_dataset = copy(self)
        new_dataset.items = new_dataset.items[start:end]
        return new_dataset

    def filter(self, filter_func):
        new_dataset = copy(self)
        new_dataset.items = [item for item in self if filter_func(item)]
        return new_dataset

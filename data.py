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
from subword-nmt import learn_bpe, apply_bpe


def list_line_locations(filename):
    line_offset = []
    offset = 0
    with open(filename) as f:
        for line in f:
            line_offset.append(offset)
            offset += len(line)
    return line_offset

class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __getitem__(self, index):
        return self.tokenizer.tokenize(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

class LinedTextDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.offsets = list_line_locations(filename)

    def __getitem__(self, index):
        with open(self.filename) as f:
            f.seek(self.offsets[index])
            item = f.readline()
        return item

    def __len__(self):
        return len(self.offsets)

class AlignedDatasets(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        items =[]
        for i, dataset in enumerate(self.datasets):
            items.append(dataset[index])
        # if all([torch.is_tensor(item) for item in items]):
            # items = torch.stack(items, )
        return items

    def __len__(self):
        return len(self.datasets[0])



check = AlignedDatasets([LinedTextDataset(d) for d in [src, target]])
# def get_chars(input_file):
#     f = open(input_file).read()
#     chars = set(f)
#     return list(chars)
#
# def simple_tokenize(captions):
#     processed = []
#     for j, s in enumerate(captions):
#         txt = str(s).lower().translate(
#             string.punctuation).strip().split()
#         processed.append(txt)
#     return processed
#
#
# def build_vocab(annFile=__TRAIN_PATH['annFile'], num_words=10000):
#     # count up the number of words
#     counts = {}
#     coco = COCO(annFile)
#     ids = coco.imgs.keys()
#     for img_id in ids:
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
#         captions = simple_tokenize([ann['caption'] for ann in anns])
#         for txt in captions:
#             for w in txt:
#                 counts[w] = counts.get(w, 0) + 1
#     cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
#
#     vocab = [w for (_, w) in cw[:num_words]]
#     vocab = [__PAD_TOKEN] + vocab + [__UNK_TOKEN, __EOS_TOKEN]
#
#     return vocab
#
#
# def create_target(vocab, rnd_caption=True):
#     word2idx = {word: idx for idx, word in enumerate(vocab)}
#     unk = word2idx[__UNK_TOKEN]
#
#     def get_caption(captions):
#         captions = simple_tokenize(captions)
#         if rnd_caption:
#             idx = randrange(len(captions))
#         else:
#             idx = 0
#         caption = captions[idx]
#         targets = []
#         for w in caption:
#             targets.append(word2idx.get(w, unk))
#         return torch.Tensor(targets)
#     return get_caption
#
#
# def create_batches(vocab, max_length=50):
#     padding = vocab.index(__PAD_TOKEN)
#     eos = vocab.index(__EOS_TOKEN)
#
#     def collate(seq):
#         seq.sort(key=lambda p: len(p[1]), reverse=True)
#         imgs, caps = zip(*seq)
#         imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)
#         lengths = [min(len(c) + 1, max_length) for c in caps]
#         batch_length = max(lengths)
#         cap_tensor = torch.LongTensor(batch_length, len(caps)).fill_(padding)
#         for i, c in enumerate(caps):
#             end_cap = lengths[i] - 1
#             if end_cap < batch_length:
#                 cap_tensor[end_cap, i] = eos
#
#             cap_tensor[:end_cap, i].copy_(c[:end_cap])
#
#         return (imgs, (cap_tensor, lengths))
#     return collate
#
#
# def get_iterator(data, batch_size=32, max_length=30, shuffle=True, num_workers=4, pin_memory=True):
#     cap, vocab = data
#     return torch.utils.data.DataLoader(
#         cap,
#         batch_size=batch_size, shuffle=shuffle,
#         collate_fn=create_batches(vocab, max_length),
#         num_workers=num_workers, pin_memory=pin_memory)

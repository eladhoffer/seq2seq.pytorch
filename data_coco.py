import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import string
from pycocotools.coco import COCO
from random import randrange
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

__COCO_IMG_PATH = "/media/ssd/Datasets/COCO/"
__COCO_ANN_PATH = "/media/ssd/Datasets/COCO/annotations/"

__TRAIN_PATH = {'root': os.path.join(__COCO_IMG_PATH, 'train2014'),
                'annFile': os.path.join(__COCO_ANN_PATH, 'captions_train2014.json')
                }
__VAL_PATH = {'root': os.path.join(__COCO_IMG_PATH, 'val2014'),
              'annFile': os.path.join(__COCO_ANN_PATH, 'captions_val2014.json')
              }

__UNK_TOKEN = 'UNK'
__PAD_TOKEN = 'PAD'
__EOS_TOKEN = 'EOS'

__normalize = {'mean': [0.485, 0.456, 0.406],
               'std': [0.229, 0.224, 0.225]}


def simple_tokenize(captions):
    processed = []
    for j, s in enumerate(captions):
        txt = str(s).lower().translate(
            string.punctuation).strip().split()
        processed.append(txt)
    return processed


def build_vocab(annFile=__TRAIN_PATH['annFile'], num_words=10000):
    # count up the number of words
    counts = {}
    coco = COCO(annFile)
    ids = coco.imgs.keys()
    for img_id in ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        captions = simple_tokenize([ann['caption'] for ann in anns])
        for txt in captions:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    vocab = [w for (_, w) in cw[:num_words]]
    vocab = [__PAD_TOKEN] + vocab + [__UNK_TOKEN, __EOS_TOKEN]

    return vocab


def create_target(vocab, rnd_caption=True):
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    unk = word2idx[__UNK_TOKEN]

    def get_caption(captions):
        captions = simple_tokenize(captions)
        if rnd_caption:
            idx = randrange(len(captions))
        else:
            idx = 0
        caption = captions[idx]
        targets = []
        for w in caption:
            targets.append(word2idx.get(w, unk))
        return torch.Tensor(targets)
    return get_caption


def create_batches(vocab, max_length=50):
    padding = vocab.index(__PAD_TOKEN)
    eos = vocab.index(__EOS_TOKEN)

    def collate(img_cap):
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps = zip(*img_cap)
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = torch.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos

            cap_tensor[:end_cap, i].copy_(c[:end_cap])

        return (imgs, (cap_tensor, lengths))
    return collate


def get_coco_data(vocab, train=True, img_size=224, scale_size=256, normalize=__normalize):
    if train:
        root, annFile = __TRAIN_PATH['root'], __TRAIN_PATH['annFile']
        img_transform = transforms.Compose([
            transforms.Scale(scale_size),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])
    else:
        root, annFile = __VAL_PATH['root'], __VAL_PATH['annFile']
        img_transform = transforms.Compose([
            transforms.Scale(scale_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])
    data = (dset.CocoCaptions(root=root, annFile=annFile, transform=img_transform,
                              target_transform=create_target(vocab, train)), vocab)
    return data


def get_iterator(data, batch_size=32, max_length=30, shuffle=True, num_workers=4, pin_memory=True):
    cap, vocab = data
    return torch.utils.data.DataLoader(
        cap,
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=create_batches(vocab, max_length),
        num_workers=num_workers, pin_memory=pin_memory)

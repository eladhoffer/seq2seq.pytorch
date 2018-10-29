import os
import logging
import string
from random import randrange
from collections import OrderedDict
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.datasets as dset
from PIL import ImageFile
from seq2seq.tools.tokenizer import Tokenizer, BPETokenizer, CharTokenizer
from seq2seq.tools import batch_sequences
from seq2seq.tools.config import EOS, BOS, PAD, LANGUAGE_TOKENS
from seq2seq.datasets.vision import create_padded_caption_batch, imagenet_transform


class CocoCaptions(object):
    """docstring for Dataset."""
    __tokenizers = {
        'word': Tokenizer,
        'char': CharTokenizer,
        'bpe': BPETokenizer
    }

    def __init__(self, root, image_transform=imagenet_transform,
                 split='train',
                 tokenization='bpe',
                 num_symbols=32000,
                 shared_vocab=True,
                 code_file=None,
                 vocab_file=None,
                 insert_start=[BOS], insert_end=[EOS],
                 mark_language=False,
                 tokenizer=None,
                 pre_tokenize=lambda x: x.lower().translate(string.punctuation),
                 sample_caption=True):
        super(CocoCaptions, self).__init__()
        self.shared_vocab = shared_vocab
        self.num_symbols = num_symbols
        self.tokenizer = tokenizer
        self.tokenization = tokenization
        self.insert_start = insert_start
        self.insert_end = insert_end
        self.mark_language = mark_language
        self.code_file = code_file
        self.vocab_file = vocab_file
        self.sample_caption = None
        self.image_transform = image_transform
        self.pre_tokenize = pre_tokenize
        if split == 'train':
            path = {'root': os.path.join(root, 'train2014'),
                    'annFile': os.path.join(root, 'annotations/captions_train2014.json')
                    }
            if sample_caption:
                self.sample_caption = randrange
        else:
            path = {'root': os.path.join(root, 'val2014'),
                    'annFile': os.path.join(root, 'annotations/captions_val2014.json')
                    }
            if sample_caption:
                self.sample_caption = lambda l: 0

        self.data = dset.CocoCaptions(root=path['root'], annFile=path[
                                      'annFile'], transform=image_transform(train=(split == 'train')))

        if self.tokenizer is None:
            prefix = os.path.join(root, 'coco')
            if tokenization not in ['bpe', 'char', 'word']:
                raise ValueError("An invalid option for tokenization was used, options are {0}".format(
                    ','.join(['bpe', 'char', 'word'])))

            if tokenization == 'bpe':
                self.code_file = code_file or '{prefix}.{lang}.{tok}.codes_{num_symbols}'.format(
                    prefix=prefix, lang='en', tok=tokenization, num_symbols=num_symbols)
            else:
                num_symbols = ''

            self.vocab_file = vocab_file or '{prefix}.{lang}.{tok}.vocab{num_symbols}'.format(
                prefix=prefix, lang='en', tok=tokenization, num_symbols=num_symbols)
            self.generate_tokenizer()

    def generate_tokenizer(self):
        additional_tokens = None
        if self.mark_language:
            additional_tokens = [LANGUAGE_TOKENS('en')]

        if self.tokenization == 'bpe':
            tokz = BPETokenizer(self.code_file,
                                vocab_file=self.vocab_file,
                                num_symbols=self.num_symbols,
                                additional_tokens=additional_tokens,
                                pre_tokenize=self.pre_tokenize)
            if not hasattr(tokz, 'bpe'):
                sentences = (d['caption']
                             for d in self.data.coco.anns.values())
                tokz.learn_bpe(sentences, from_filenames=False)
        else:
            tokz = self.__tokenizers[self.tokenization](
                vocab_file=self.vocab_file,
                additional_tokens=additional_tokens,
                pre_tokenize=self.pre_tokenize)

        if not hasattr(tokz, 'vocab'):
            sentences = (d['caption'] for d in self.data.coco.anns.values())
            logging.info('generating vocabulary. saving to %s' %
                         self.vocab_file)
            tokz.get_vocab(sentences, from_filenames=False)
            tokz.save_vocab(self.vocab_file)
        self.tokenizer = tokz

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        img, captions = self.data[index]
        insert_start = self.insert_start
        insert_end = self.insert_end

        def transform(t):
            return self.tokenizer.tokenize(t,
                                           insert_start=insert_start,
                                           insert_end=insert_end)
        if self.sample_caption is None:
            captions = [transform(c) for c in captions]
        else:
            captions = transform(
                captions[self.sample_caption(len(captions))])
        return (img, captions)

    def __len__(self):
        return len(self.data)

    def get_loader(self, batch_size=1, shuffle=False, pack=False, sampler=None, num_workers=0,
                   max_length=100, max_tokens=None, batch_first=False,
                   pin_memory=False, drop_last=False, augment=False):
        collate_fn = create_padded_caption_batch(
            max_length=max_length, max_tokens=max_tokens,
            pack=pack, batch_first=batch_first, augment=augment)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)

    @property
    def tokenizers(self):
        return OrderedDict(img=self.image_transform, en=self.tokenizer)

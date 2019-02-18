import os
import json
import logging
import string
from random import randrange
from collections import OrderedDict
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import ImageFile
from seq2seq.tools.tokenizer import Tokenizer, BPETokenizer, CharTokenizer
from seq2seq.tools import batch_sequences
from seq2seq.tools.config import EOS, BOS, PAD, LANGUAGE_TOKENS
from seq2seq.datasets import LinedTextDataset
from seq2seq.datasets.vision import create_padded_caption_batch, imagenet_transform


def get_defected_list(item_list, callback):
    defected = []
    for i, item in enumerate(item_list):
        try:
            callback(item)
        except:
            defected.append(i)
    return defected


class ConceptCaptions(object):
    """docstring for Dataset."""
    __tokenizers = {
        'word': Tokenizer,
        'char': CharTokenizer,
        'bpe': BPETokenizer
    }

    def __init__(self, root, image_transform=imagenet_transform,
                 split='train',
                 tokenization='word',
                 shared_vocab=True,
                 tokenization_config={},
                 tokenization_model_file=None,
                 insert_start=[BOS], insert_end=[EOS],
                 mark_language=False,
                 tokenizer=None,
                 pre_tokenize=None,
                 vocab_limit=None,
                 vocab_min_count=2,
                 loader=default_loader):
        super(ConceptCaptions, self).__init__()
        self.split = split
        self.shared_vocab = shared_vocab
        self.tokenization_config = tokenization_config
        self.tokenizer = tokenizer
        self.tokenization = tokenization
        self.insert_start = insert_start
        self.insert_end = insert_end
        self.mark_language = mark_language
        self.vocab_limit = vocab_limit
        self.vocab_min_count = vocab_min_count
        if image_transform is not None:
            self.transform = image_transform(train=(split == 'train'))
        else:
            self.transform = None
        self.pre_tokenize = pre_tokenize
        self.loader = loader
        if split == 'train':
            path = {'root': os.path.join(root, 'training'),
                    'annFile': os.path.join(root, 'training.txt'),
                    'filtered': os.path.join(root, 'defected_training.json')
                    }
        else:
            path = {'root': os.path.join(root, 'validation'),
                    'annFile': os.path.join(root, 'validation.txt'),
                    'filtered': os.path.join(root, 'defected_validation.json')
                    }
        self.image_path = path['root']
        self.captions = LinedTextDataset(path['annFile'])
        if os.path.isfile(path['filtered']):
            with open(path['filtered'], 'r') as f:
                filtered = json.loads(f.read())
        else:
            filtered = get_defected_list(range(len(self.captions)),
                                         lambda idx: self._load_image(idx))
            with open(path['filtered'], 'w') as f:
                f.write(json.dumps(filtered))

        self.indexes = list(set(range(len(self.captions))) - set(filtered))

        if tokenization == 'bpe' or tokenization == 'sentencepiece':
            self.tokenization_config.setdefault('num_symbols', 32000)

        tok_setting = [tokenization]
        for n, v in tokenization_config.items():
            if (isinstance(v, bool) and v):
                tok_setting.append(n)
            elif n == 'model_type' or n == 'num_symbols':
                tok_setting.append(str(v))
            else:
                tok_setting.append('%s-%s' % (n, v))

        tok_prefix = '.'.join(tok_setting)

        if self.tokenizer is None:
            if tokenization not in ['sentencepiece', 'bpe', 'char', 'word', 'word+char']:
                raise ValueError("An invalid option for tokenization was used, options are {0}".format(
                    ','.join(['sentencepiece', 'bpe', 'char', 'word'])))

            self.tok_file = tokenization_model_file or '{prefix}.{lang}'.format(
                prefix='captions', tok=tok_prefix, lang='en')

            self.generate_tokenizer()

    def generate_tokenizer(self):
        additional_tokens = None
        if self.mark_language:
            additional_tokens = [LANGUAGE_TOKENS('en')]

        if self.tokenization == 'bpe':
            tokz = BPETokenizer(self.tok_file, **self.tokenization_config)
            if not hasattr(tokz, 'bpe'):
                sentences = (self.captions[i] for i in self.indexes)
                tokz.learn_bpe(sentences, from_filenames=False)
        else:
            tokz = self.__tokenizers[self.tokenization](
                self.tok_file, **self.tokenization_config)

        if not hasattr(tokz, 'vocab'):
            assert self.split == 'train', "better generate vocab for training split"
            sentences = (self.captions[i] for i in self.indexes)
            logging.info('generating vocabulary. saving to %s' %
                         self.tok_file)
            tokz.get_vocab(sentences, from_filenames=False)
            tokz.save_vocab(self.tok_file)
        tokz.load_vocab(self.tok_file, limit=self.vocab_limit,
                        min_count=self.vocab_min_count)
        self.tokenizer = tokz

    def _load_image(self, index):
        return self.loader('{}/{}.jpg'.format(self.image_path, str(index)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        index = self.indexes[index]
        img = self._load_image(index)
        if self.transform is not None:
            img = self.transform(img)
        caption = self.tokenizer.tokenize(self.captions[index],
                                          insert_start=self.insert_start,
                                          insert_end=self.insert_end)

        return (img, caption)

    def __len__(self):
        return len(self.indexes)

    def get_loader(self, batch_size=1, shuffle=False, pack=False, sampler=None, num_workers=0,
                   max_length=None, fixed_length=None, batch_first=False,
                   pin_memory=False, drop_last=False, augment=False):
        collate_fn = create_padded_caption_batch(
            max_length=max_length, 
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
        return OrderedDict(img=self.transform, en=self.tokenizer)


if __name__ == '__main__':
    data = ConceptCaptions(
        '/media/drive/Datasets/concept_captions', split='train', image_transform=None)


# #Now read the file back into a Python list object
# with open('test.txt', 'r') as f:
#     a = json.loads(f.read())

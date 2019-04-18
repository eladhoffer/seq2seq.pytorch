import torch
import torchvision.transforms as transforms
from PIL import ImageFile
from seq2seq.tools import batch_sequences
from seq2seq.tools.config import EOS, BOS, PAD, LANGUAGE_TOKENS
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Palette images with Transparency", UserWarning)

def imagenet_transform(scale_size=256, input_size=224, train=True, augmentation='inception', allow_var_size=False):
    normalize = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}

    if train:
        if augmentation == 'inception':
            return transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.5, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(scale_size),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)
            ])
    elif allow_var_size:
        return transforms.Compose([
            transforms.Resize(scale_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])


def create_padded_caption_batch(max_length=100, max_tokens=None, batch_first=False,
                                sort=False, pack=False, augment=False):
    def collate(img_seq_tuple):
        if sort or pack:  # packing requires a sorted batch by length
            img_seq_tuple.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, seqs = zip(*img_seq_tuple)
        imgs = torch.stack(imgs, 0)
        seq_tensor = batch_sequences(seqs, max_length=max_length,
                                     max_tokens=max_tokens,
                                     batch_first=batch_first,
                                     sort=False, pack=pack, augment=augment)
        return (imgs, seq_tensor)
    return collate

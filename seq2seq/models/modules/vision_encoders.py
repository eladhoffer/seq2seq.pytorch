import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet, densenet, vgg, alexnet, squeezenet


class CNNEncoderBase(nn.Module):
    """docstring for CNNEncoder."""

    def __init__(self, finetune=True):
        super(CNNEncoderBase, self).__init__()
        self.finetune = finetune
        self.batch_first = True
        self.toggle_grad()

    def toggle_grad(self):
        requires_grad = self.finetune
        self.finetune = True  # to access params
        for p in self.parameters():
            p.requires_grad = requires_grad
        self.finetune = requires_grad  # to deny access params

    def named_parameters(self, memo=None, prefix=''):
        if self.finetune:
            return super(CNNEncoderBase, self).named_parameters(memo, prefix)
        else:
            return set()

    def state_dict(self, destination=None, prefix=''):
        if self.finetune:
            return super(CNNEncoderBase, self).state_dict(destination, prefix)
        else:
            return ()

    def load_state_dict(self, state_dict):
        if self.finetune:
            return super(CNNEncoderBase, self).load_state_dict(state_dict)


class ResNetEncoder(CNNEncoderBase):

    def __init__(self, model='resnet50', pretrained=True, finetune=True):
        super(ResNetEncoder, self).__init__(finetune=finetune)
        self.model = resnet.__dict__[model](pretrained=pretrained)
        self.hidden_size = self.model.fc.in_features
        del self.model.fc

    def forward(self, x):
        self.toggle_grad()
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class DenseNetEncoder(CNNEncoderBase):

    def __init__(self, model='densenet121', pretrained=True, finetune=True):
        super(DenseNetEncoder, self).__init__(finetune=finetune)
        self.model = densenet.__dict__[model](pretrained=pretrained)
        self.hidden_size = self.model.classifier.in_features
        del self.model.classifier

    def forward(self, x):
        self.toggle_grad()
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7)
        return out


class VGGEncoder(CNNEncoderBase):

    def __init__(self, model='vgg16', pretrained=True, finetune=True):
        super(VGGEncoder, self).__init__(finetune=finetune)
        self.model = vgg.__dict__[model](pretrained=pretrained)
        self.hidden_size = self.model.classifier.in_features
        del self.model.classifier

    def forward(self, x):
        self.toggle_grad()
        x = self.features(x)
        return x


class AlexNetEncoder(CNNEncoderBase):

    def __init__(self, model='alexnet', pretrained=True, finetune=True):
        super(AlexNetEncoder, self).__init__(finetune=finetune)
        self.model = alexnet.__dict__[model](pretrained=pretrained)
        self.hidden_size = self.model.classifier.in_features
        del self.model.classifier

    def forward(self, x):
        self.toggle_grad()
        x = self.features(x)
        return x


class SqueezeNetEncoder(CNNEncoderBase):

    def __init__(self, model='squeezenet1_1', pretrained=True, finetune=True):
        super(SqueezeNetEncoder, self).__init__(finetune=finetune)
        self.model = squeezenet.__dict__[model](pretrained=pretrained)
        self.hidden_size = 512
        del self.model.classifier

    def forward(self, x):
        self.toggle_grad()
        x = self.features(x)
        return x

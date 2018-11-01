import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet, densenet, vgg, alexnet, squeezenet


class CNNEncoderBase(nn.Module):
    """docstring for CNNEncoder."""

    def __init__(self, model, context_size, context_transform=None, context_nonlinearity=None, spatial_context=True, finetune=True):
        super(CNNEncoderBase, self).__init__()
        self.model = model
        self.finetune = finetune
        self.batch_first = True
        self.toggle_grad()
        self.spatial_context = spatial_context
        if context_transform is None:
            self.context_size = context_size
        else:
            if self.spatial_context:
                self.context_transform = nn.Conv2d(
                    context_size, context_transform, 1)
            else:
                self.context_transform = nn.Linear(
                    context_size, context_transform)
            if context_nonlinearity is not None:
                self.context_nonlinearity = F.__dict__[context_nonlinearity]
            self.context_size = context_transform

    def toggle_grad(self):
        for p in self.model.parameters():
            p.requires_grad = self.finetune

    def named_parameters(self, *kargs, **kwargs):
        if self.finetune:
            return super(CNNEncoderBase, self).named_parameters(*kargs, **kwargs)
        elif hasattr(self, 'context_transform'):
            return self.context_transform.named_parameters(*kargs, **kwargs)
        else:
            return set()

    def state_dict(self, *kargs, **kwargs):
        if self.finetune:
            return super(CNNEncoderBase, self).state_dict(*kargs, **kwargs)
        elif hasattr(self, 'context_transform'):
            return self.context_transform.state_dict(*kargs, **kwargs)
        else:
            return {}

    def load_state_dict(self, state_dict, *kargs, **kwargs):
        if self.finetune:
            return super(CNNEncoderBase, self).load_state_dict(state_dict, *kargs,  **kwargs)
        elif hasattr(self, 'context_transform'):
            return self.context_transform.load_state_dict(state_dict, *kargs,  **kwargs)
        else:
            return


class ResNetEncoder(CNNEncoderBase):

    def __init__(self, model='resnet50', pretrained=True, **kwargs):
        model = resnet.__dict__[model](pretrained=pretrained)
        super(ResNetEncoder, self).__init__(model,
                                            context_size=model.fc.in_features,
                                            **kwargs)
        del self.model.fc

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

        if not self.spatial_context:
            x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), x.size(1))
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class DenseNetEncoder(CNNEncoderBase):

    def __init__(self, model='densenet121', pretrained=True, **kwargs):
        model = densenet.__dict__[model](pretrained=pretrained)
        super(DenseNetEncoder, self).__init__(model, context_size=model.classifier.in_features,
                                              **kwargs)
        del self.model.classifier

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            features = self.model.features(x)
            x = F.relu(features, inplace=True)
        if not self.spatial_context:
            x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), x.size(1))
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class VGGEncoder(CNNEncoderBase):

    def __init__(self, model='vgg16', pretrained=True, **kwargs):
        model = vgg.__dict__[model](pretrained=pretrained)
        super(VGGEncoder, self).__init__(model, context_size=model.classifier.in_features,
                                         **kwargs)
        del self.model.classifier

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            x = self.features(x)
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class AlexNetEncoder(CNNEncoderBase):

    def __init__(self, model='alexnet',  pretrained=True, **kwargs):
        model = alexnet.__dict__[model](pretrained=pretrained)
        super(AlexNetEncoder, self).__init__(model, context_size=model.classifier.in_features,
                                             **kwargs)
        del self.model.classifier

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            x = self.features(x)
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x


class SqueezeNetEncoder(CNNEncoderBase):

    def __init__(self, model='squeezenet1_1', pretrained=True, **kwargs):
        model = squeezenet.__dict__[model](pretrained=pretrained)
        super(SqueezeNetEncoder, self).__init__(model, context_size=model.classifier.in_features,
                                             **kwargs)
        del self.model.classifier

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune):
            x = self.features(x)
        if hasattr(self, 'context_transform'):
            x = self.context_transform(x)
        if hasattr(self, 'context_nonlinearity'):
            x = self.context_nonlinearity(x)
        return x

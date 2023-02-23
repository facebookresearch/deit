import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, BatchNorm2d, MaxPool2d, AvgPool2d

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, use_bn=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], use_bn)
        self.classifier = Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, use_bn):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2d(in_channels, x, kernel_size=3, padding=1)]
                if use_bn:
                    layers += [BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        layers += [AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(use_bn=True):
    return VGG('VGG11', use_bn)
def VGG13(use_bn=True):
    return VGG('VGG13', use_bn)
def VGG16(use_bn=True):
    return VGG('VGG16', use_bn)
def VGG19(use_bn=True):
    return VGG('VGG19', use_bn)
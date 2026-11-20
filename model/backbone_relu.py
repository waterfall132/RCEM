import torch
import torch.nn as nn
from torch.nn import init

import models.backbone as backbone


class Conv64F_Local_ReLU(nn.Module):

    def __init__(self):
        super(Conv64F_Local_ReLU, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.features(x)


def _relu_patch(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(module, name, nn.ReLU(inplace=child.inplace))
        else:
            _relu_patch(child)
    return module


def ResNet12_ReLU(**kwargs):
    net = backbone.ResNet12(**kwargs)
    return _relu_patch(net)


def SeResNet12_ReLU(**kwargs):
    net = backbone.SeResNet12(**kwargs)
    return _relu_patch(net)


@torch.no_grad()
def negative_fraction(feat):
    return (feat < 0).float().mean().item()

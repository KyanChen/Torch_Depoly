from .resnet import get_resnet

import torch

def build_backbone(name='resnet50',**kwargs):
    if name.startswith('resnet'):
        return get_resnet(name,**kwargs)
    else:
        raise NotImplementedError
import os

import segmentation_models_pytorch as smp
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def get_unet(encoder: str ='resnet50', in_channels: int = 4, num_classes: int = 1, activation: str = None):
    """
    Get Unet model from qubvel libruary
    create segmentation model with a pretrained encoder
    Args: 
        encoder (str): encoder basenet 'resnext101_32x8d', 'resnet18', 'resnet50', 'resnet101'...
        in_channels (int): bnumber of input channels
        num_classes (int): number of classes + 1 for background        
        activation (srt): output activation function, default is None
    """
    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights='imagenet',
        classes=num_classes, 
        in_channels=in_channels,
        activation=activation,)

    return model

model = smp.FPN('resnet34', in_channels=1)


def get_fpn(encoder: str='resnet50', in_channels: int = 4, num_classes: int = 1, activation: str = None):
    """
    Get FPN model from qubvel libruary
    create segmentation model with pretrained encoder
    Args: 
        encoder (str): encoder basenet 'resnext101_32x8d', 'resnet18', 'resnet50', 'resnet101'...
        in_channels (int): bnumber of input channels
        num_classes (int): number of classes + 1 for background        
        activation (srt): output activation function, default is None
    """
    model = smp.FPN(
        encoder_name=encoder, 
        encoder_weights='imagenet',
        classes=num_classes, 
        in_channels=in_channels,
        activation=activation,)

    return model    


def _recompile_intro(self):
    """
    change model input for 4 challens with the pretrain
    """
    weight1 = self.encoder.block1[0].weight.data
    bias1 = self.encoder.block1[0].bias.data

    self.encoder.block1 = nn.Sequential(
            conv3x3(self.in_channels, self.num_filters), self.encoder.block1[1], self.encoder.block1[2],
            self.encoder.block1[3]
        )
    self.encoder.block1[0].weight.data = torch.cat([weight1 / 2, weight1 / 2], 1)
    self.encoder.block1[0].bias.data = bias1

      
def print_model_summary(model):
    """Prints all layers and dims of the pytorch net"""    
    import torchsummary
    torchsummary.summary(model, (1, 512, 512))
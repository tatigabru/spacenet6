"""
Soft Jaccard Loss

Adapted from:
    Source: https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/
    Author: E. Khvedchenya (BloodAxe)
"""
from typing import List

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from pytorch_toolbelt.utils.torch_utils import to_tensor
from torch import Tensor
from torch.nn.modules.loss import _Loss

__all__ = ["JaccardLoss"]


class JaccardLoss(_Loss):
    """
    Implementation of soft Jaccard loss for binary image segmentation task    
    """

    def __init__(self, log_loss: bool=True, log_sigmoid: bool=True, smooth: float=0.0):

        super(JaccardLoss, self).__init__
        self.log_loss = log_loss
        self.log_sigmoid = log_sigmoid
        self.smooth = smooth
                
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Args:
            y_pred: binary masks of shape Nx1xHxW
            y_true: binary masks of shape Nx1xHxW        
        """
        if y_true.size() != y_pred.size(): raise ValueError("Model outputs and targetd mush have the same size: Nx1xHxW") 
        batch_size = y_true.size(0) # batch size
        eps = 1e-7

        if self.log_sigmoid:            
            # Log-Exp gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            outputs = F.logsigmoid(outputs).exp()
        else:
            outputs = F.sigmoid(outputs)        
        targets = (y_true == 1).type_as(outputs)

        intersection = torch.sum(outputs * targets)
        union = torch.sum(outputs + targets) - intersection
        jaccard_loss = (intersection + self.smooth) / (union.clamp_min(eps) + self.smooth) 

        if self.log_loss:
            jaccard_loss = -torch.log(jaccard_loss)
        else: 
            jaccard_loss = 1 - jaccard_loss   

        return jaccard_loss 

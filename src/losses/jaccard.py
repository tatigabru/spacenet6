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
from torch import Tensor
from torch.nn.modules.loss import _Loss

__all__ = ["JaccardLoss"]


class JaccardLoss(_Loss):
    """
    Implementation of soft Jaccard loss for binary image segmentation task  
    calculated from logits   
        Args: 
            log_loss: if True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
            log_sigmoid: if True, uses F.logsigmoid(y_pred).exp(), otherwise uses torch.sigmoid(y_pred)  
            smooth: smooth the loss values, should be float between 0 and 1
            reduction: str = 'mean', 'sum' or 'none'. Default = 'mean'
        Output: loss as a float or Tensor (for reduction 'none')  

    """

    def __init__(self, log_sigmoid: bool=True, log_loss: bool=False, smooth: float=0.0, reduction: str = 'mean'):

        super(JaccardLoss, self).__init__()
        
        self.log_sigmoid = log_sigmoid
        self.log_loss = log_loss
        self.smooth = smooth
        self.reduction = reduction
                
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Args:
            y_pred: binary masks of shape Nx1xHxW
            y_true: binary masks of shape Nx1xHxW        
        """
        if y_true.size() != y_pred.size(): raise ValueError("Model outputs and targetd mush have the same size: Nx1xHxW")         
        eps = 1e-7

        if self.log_sigmoid:            
            # Log-Exp more numerically stable and does not cause vanishing gradient on
            # extreme values of 0 and 1
            outputs = F.logsigmoid(y_pred).exp()
        else:
            #outputs = y_pred    
            outputs = torch.sigmoid(y_pred)        
        targets = (y_true == 1).type_as(outputs) # ensure the same type

        intersection = torch.sum(outputs * targets)
        union = torch.sum(outputs + targets) - intersection
        jaccard_loss = (intersection + self.smooth) / (union.clamp_min(eps) + self.smooth) 

        if self.log_loss:
            jaccard_loss = -torch.log(jaccard_loss.clamp_min(eps))
        else: 
            jaccard_loss = 1 - jaccard_loss   

        if self.reduction == "mean":
            jaccard_loss = jaccard_loss.mean()
        elif self.reduction == "sum":
            jaccard_loss = jaccard_loss.sum()     

        return jaccard_loss


def test_loss():
    size = (2,1,2,2)
    # uniform distrubution of number in -2 .. +2 range
    y_pred = 4*torch.rand(size)-2 # uniform distrubution of number in -2 .. +2 range
    y_true = torch.randint(low = 0, high = 2, size = size).type_as(y_pred)
    print(f'y_pred \n {y_pred}')
    print(f'torch.sigmoid(y_pred) \n {torch.sigmoid(y_pred)}')
    print(f'y_true \n {y_true}')

    criterion = JaccardLoss(log_sigmoid=True)
    loss = criterion(y_pred, y_true)
    print(f'JaccardLoss() {loss}') 
    
    criterion = JaccardLoss(log_sigmoid=False)
    loss = criterion(y_true, y_true)
    print(f'(y_true, y_true) JaccardLoss(log_sigmoid=False) {loss}')      
    

if __name__ == "__main__":
    test_loss()

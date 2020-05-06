"""
BCE - Jaccard Loss

Adapted from:
 
    Source: https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/
    Author: E. Khvedchenya (BloodAxe)
    Source: https://github.com/ternaus/robot-surgery-segmentation/blob/master/loss.py
    Author: V.Iglovikov (ternaus)
    
"""
from typing import List
from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_toolbelt.utils.torch_utils import to_tensor
from torch import Tensor
from torch.nn.modules.loss import _Loss

__all__ = ["BCEJaccardLoss"]


class BCEJaccardLoss(_Loss):
    """
    Implementation of combination of BCE and soft Jaccard loss 
    for binary image segmentation task, calculated from logits 

    """

    def __init__(self, bce_weight: float = 0.5, jaccard_weight: float = 0.5, log_loss: bool=False, 
                 log_sigmoid: bool = True, smooth: float=0.0, reduction: str = 'mean'):

        super(BCEJaccardLoss, self).__init__()
        self.bce_weight = bce_weight
        self.jaccard_weight = jaccard_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction) # from logits
        self.log_sigmoid = log_sigmoid
        self.log_loss = log_loss        
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:   
        """
        Returns forward propagation of the loss
        
        Args:
            y_pred: binary masks of shape Nx1xHxW
            y_true: binary masks of shape Nx1xHxW        
        """
        if y_true.size() != y_pred.size(): raise ValueError("Model outputs and targetd mush have the same size: Nx1xHxW") 

        loss = self.bce_weight*self.bce_loss(y_pred, y_true)

        if self.jaccard_weight != 0:
            eps = 1e-7
            if self.log_sigmoid:            
                # Log-Exp gives more numerically stable result and does not cause vanishing gradient on
                # extreme values 0 and 1
                outputs = F.logsigmoid(y_pred).exp()
            else:
                outputs = torch.sigmoid(y_pred)  
            targets = (y_true == 1).type_as(outputs)

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

            loss += self.jaccard_weight*jaccard_loss     

        return loss 


def test_loss():
    size = (2,1,4,4)
    # uniform distrubution of number in -2 .. +2 range
    y_pred = 4*torch.rand(size)-2 # uniform distrubution of number in -2 .. +2 range
    y_true = torch.randint(low = 0, high = 2, size = size).type_as(y_pred)
    print(f'y_pred {y_pred}')
    print(f'torch.sigmoid(y_pred) {torch.sigmoid(y_pred)}')
    print(f'y_true {y_true}')

    criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
    loss_bce = criterion(y_pred, y_true)  

    criterion = BCEJaccardLoss(bce_weight=1, jaccard_weight=0, log_sigmoid=True)
    loss = criterion(y_pred, y_true)
    print(f'BCEJaccardLoss(bce_weight=1, jaccard_weight=0, log_sigmoid=True) {loss}') 
    
    criterion = BCEJaccardLoss(bce_weight=0, jaccard_weight=1)
    loss = criterion(y_pred, y_true)
    print(f'BCEJaccardLoss(bce_weight=0, jaccard_weight=1, log_sigmoid=True) {loss}')  
    
    criterion = BCEJaccardLoss(bce_weight=0.5, jaccard_weight=0.5, log_loss=False)
    loss = criterion(y_pred, y_true)
    print(f'BCEJaccardLoss(bce_weight=0.5, jaccard_weight=0.5, log_sigmoid=True) {loss}')  
    
    criterion = BCEJaccardLoss(bce_weight=0.5, jaccard_weight=0.5, log_loss=True)
    loss = criterion(y_pred, y_true)
    print(f'BCEJaccardLoss(bce_weight=0.5, jaccard_weight=0.5, log_loss=True) {loss}')  
   

if __name__ == "__main__":
    test_loss()
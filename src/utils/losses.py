"""
Custom losses from: 

https://github.com/tugstugi/pytorch-saltnet/blob/master/losses/lovasz_losses.py

"""
import os 
import math
import numpy as np
import torch
from sklearn.utils import compute_sample_weight


def get_balanced_weights(masks_dir):
    all_masks = os.glob
    labels=[]
    for mask in dataset.masks_fps:
      mask = fs.read_image_as_is(mask)
      unique_labels = np.unique(mask)
      labels.append(''.join([str(int(i)) for i in unique_labels]))

    weights = compute_sample_weight('balanced', labels)
    return weights


def dice_loss(pred: np.array, target: np.array):
    smooth = 1.

    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


def dice_loss_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return 1 - (numerator + 1) / (denominator + 1)      


def binarize_predictions(prediction: np.array) -> np.array:
    """Binarise output masksfor one-hot classes"""
    #prediction = prediction.detach().cpu().numpy()
    output = np.rint(prediction).astype(np.uint8)
    output_one_hot = (output[:, :, None] -1 == np.arange(num_classes)[None, None, :]).astype(np.uint8)
    
    return output_one_hot

def test_dice_loss() -> None:
    a = np.array([[1.1, 1.4, 5.3, 4.7],
                 [2.1, 4.6, 1.3, 8.9], 
                 [7.3, 3.4, 3.3, 3.2],
                 [7.1, 7.6, 8.3, 8.9]]) 

    b = np.array([[1, 1, 5, 4],
                 [2, 2, 3, 9],
                 [7, 3, 3, 3],
                 [7, 2, 8, 9]]) 
    print(f'a {a},\n b {b}') 
    a = np.rint(a).astype(np.uint8)         
    print(dice_loss(a, b))


if __name__ == "__main__":
   test_dice_loss()        





    

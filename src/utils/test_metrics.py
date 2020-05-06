import torch 
import numpy as np
import torch.nn.functional as F

from iou import *

SMOOTH = 1e-6
NUM_CLASSES = 5


def test_mean_iou():
    A = np.array([[1.1, 1.4, 5.3, 0],
                 [2.1, 4.6, 2.3, 0], 
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]) 

    B = np.array([[1, 1, 5, 4],
                 [2, 2, 1, 1],
                 [7, 3, 3, 3],
                 [0, 0, 0, 0]])
    A = torch.Tensor(A)
    B = torch.Tensor(B)
    #A = torch.Tensor(A).cuda().round().type(torch.int8)
    mIoU = mean_iou(A, B)
    print(mIoU)


def test_pix_acc():
    A = np.array([[1.1, 1.4, 5.3, 0],
                 [2.1, 4.6, 2.3, 0], 
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]) 

    B = np.array([[1, 1, 5, 4],
                 [2, 2, 1, 1],
                 [7, 3, 3, 3],
                 [0, 0, 0, 0]])
    A = torch.Tensor(A).round().type(torch.int8)
    B = torch.Tensor(B).round().type(torch.int8)
    #A = torch.Tensor(A).cuda().round().type(torch.int8)
    pix_acc = pixel_acc(A, B)
    print(pix_acc)


def main():
    test_mean_iou()
    test_pix_acc()
 
    


if __name__ == "__main__":
    main()        
 


def test_iou_pytorch():
    a = np.array([[1.1, 1.4, 5.3, 0],
                 [2.1, 4.6, 2.3, 0], 
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]) 

    b = np.array([[1, 1, 5, 4],
                 [2, 2, 1, 1],
                 [7, 3, 3, 3],
                 [0, 0, 0, 0]])
    a_rounded = torch.Tensor(a).cuda().round().type(torch.int8)
    a_rounded= a_rounded.dtype(torch.int8)
    print(f'a {a}\n b {b} \n a_rounded {a_rounded}, a_rounded.dtype {a_rounded.dtype}') 
    ios, iou = iou_pytorch(a, b)
    print(f"IoUs: {ious} mean IoU: {iou}")


def test_iou_multiclass():
    A = torch.tensor([
    [[1.1, 1.2, 2.3, 2.1],
     [1, 1, 2, 3],
     [1, 1, 3, 3]]
    ]) 
    B = torch.tensor([
    [[1, 1, 2, 2],
     [1, 1, 2, 2],
     [1, 3, 3, 2]]
    ])
    A = torch.Tensor(A).cuda().round().type(torch.int8)
    A_oh = F.one_hot(A)
    B_oh = F.one_hot(B)
    print(f'a {A_oh},\n b {A_oh}')

    intersection = np.logical_and(A_oh, B_oh).sum(1).sum(1).type(torch.float32)
    union = np.logical_or(A_oh, B_oh).sum(1).sum(1).type(torch.float32)
    iou = intersection / union
    print(iou[:, 1:])


def test_iou_sample():
    a = np.array([[1.1, 1.4, 5.3, 0],
                 [2.1, 4.6, 2.3, 0], 
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]) 

    b = np.array([[1, 1, 5, 4],
                 [2, 2, 1, 1],
                 [7, 3, 3, 3],
                 [0, 0, 0, 0]]) 
    print(f'a {a},\n b {b}')             

    a = np.rint(a).astype(np.uint8)
    ious, iou = iou_numpy(a,b) 
    rint(f"IoUs: {ious} mean IoU: {iou}")

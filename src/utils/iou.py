import torch 
import numpy as np
import torch.nn.functional as F
from torch import Tensor

SMOOTH = 1e-7
NUM_CLASSES = 2


def binary_iou_numpy(output: torch.Tensor, target: torch.Tensor, from_logits: bool = True) -> float:
    """
    Batch images IoUs for a  binary segmentation  
    Args: 
        output: model output as a torch.Tensor 
        target: true masks as a torch.Tensor
    """    
    if from_logits:
        # binary output
        output = torch.sigmoid(output)
        output.round() #output.cuda().round()
        
    output = np.asarray(output.detach().cpu().numpy())
    target = np.asarray(target.detach().cpu().numpy())
    if output.ndim == 4:
        output = np.squeeze(output, axis=1)  # BATCH x 1 x H x W => BATCH x H x W
    if target.ndim == 4:
        target = np.squeeze(target, axis=1)
    # Remove background
    target = (target == 1)
    # Compute area intersection:
    intersection = (output * target).sum()
    # Union, sums ones excluding batch dimension
    union = output.sum() + target.sum() - intersection 
    iou = intersection / (union + SMOOTH)  
    
    return iou.mean()


def binary_iou_pytorch(output: Tensor, target: Tensor, from_logits: bool = True) -> float:
    #  BATCH x 1 x H x W shape
    if output.ndim == 4:
        output = output.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    if target.ndim == 4:    
        target = target.squeeze(1)
    # Remove background
    target = (target == 1).type_as(output)
    if from_logits:
        # binary output
        output = torch.sigmoid(output)
        output.round() #output.cuda().round()
    # Compute area intersection:
    intersection = torch.sum(output * target)
    # Union, sums ones excluding batch dimension
    union = torch.sum(output + target) - intersection 
    iou = intersection / (union + SMOOTH)  
   
    return iou.mean() 


def test_iou():
    output = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]).unsqueeze(0)
    target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]).unsqueeze(0)
    print(output, target)
    iou = binary_iou_numpy(output, target)
    print(f'iou numpy: {iou}')
    assert abs(iou - 0.5) < 0.01
    iou = binary_iou_pytorch(output, target)
    print(f'iou pytorch: {iou}')
    assert abs(iou - 0.5) < 0.01

    
def official_pixel_acc(output: torch.tensor, target: torch.tensor) -> tuple:
    """
    Official pixel accuracy
    as from
    https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
    # This function takes the prediction and label of a single image, returns pixel-wise accuracy
    # To compute over many images do:
    # for i = range(Nimages):
    #	(pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = pixelAccuracy(imPred[i], imLab[i])
    # mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    output = np.asarray(output)
    target = np.asarray(target)
	# Remove classes from unlabeled pixels in gt image	
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output==target)*(target > 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

    return (pixel_accuracy, pixel_correct, pixel_labeled)

    
def pixel_acc(output: torch.tensor, target: torch.tensor, ignore_index:int = 0) -> float:
    """Calculate pixel accuracy between model prediction and target
       Args:
           output (torch.tensor): model output of shape (B,C,H,W);
                each class in one channel (C == n_classes)
           target (torch.tensor): target tensor of shape (B,H,W);
                class objects encoded by unique values        
    """
    mask = (target != ignore_index)
    #output = torch.argmax(output, dim=1)
    correct = (target == output)
    accuracy = (correct * mask).sum().float() / mask.sum()
    
    return accuracy


def pytorch_iou(output: torch.tensor, target: torch.tensor, num_class: int = NUM_CLASSES) -> tuple:
    """
    PyTorch metric from official metric as in 
    https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
    """
	# Remove classes from unlabeled pixels in gt image
    output = output * (target > 0)
    # Compute area intersection:
    intersection = output * (output == target)
    area_intersection = torch.histc(intersection, bins=num_class, min=1, max=num_class)
    # Compute area union
    area_pred = torch.histc(output, bins=num_class, min=1, max=num_class)
    area_lab = torch.histc(target, bins=num_class, min=1, max=num_class)
    area_union = area_pred + area_lab - area_intersection
        
    return (area_intersection, area_union)
    

def official_iou(output: torch.tensor, target: torch.tensor, numClass: int = NUM_CLASSES) -> tuple:
    """
    Official metric from 
    https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
    
    This function takes the prediction and label of a single image, returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
    	(area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    #output = torch.argmax(output, dim=1)
    output = np.asarray(output.detach().cpu().numpy())
    target = np.asarray(target.detach().cpu().numpy())
	# Remove classes from unlabeled pixels in gt image.We should not penalize detections in unlabeled portions 
    output = output * (target > 0)
	# Compute area intersection:
    intersection = output * (output == target)
    (area_intersection,_) = np.histogram(intersection, bins=numClass, range=(1, numClass))
	# Compute area union:
    (area_pred,_) = np.histogram(output, bins=numClass, range=(1, numClass))
    (area_lab,_) = np.histogram(target, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    
    return (area_intersection, area_union)    


def _ignore_channels(*xs, ignore_channels=None, dim=1):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[dim]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=dim, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def mean_iou(output: np.array, target: np.array) -> float:
    """Calculate mean IoU score averaged for a batch
       Args:
           output (torch.tensor): model output of shape (B,C,H,W);
                each class in one channel (C == n_classes)
           target (torch.tensor): target tensor of shape (B,H,W);
                class objects encoded by unique values 
           ignore_empty (bool): if True, classes which are not presented on target mask,
                and not predicted by `net` are not included to metric calculation        
    """
    n_classes = output.shape[1]
    # prepare output
    #output = torch.argmax(output, dim=1)
    # Remove classes from unlabeled pixels in gt image
    output = output * (target > 0)
    # convert target to onehot BHWC
    target = F.one_hot(target, n_classes)
    output = F.one_hot(output, n_classes)    
    # intersection-over-union
    intersection = (output * target).sum(dim=(0, 1, 2)).float()
    union = (output + target).sum(dim=(0, 1, 2)).float() - intersection
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    return iou.mean()


def cm(pred: torch.tensor, true: torch.tensor) -> float:
    """
    Confusion matrix
    """
    c = pred.shape[1]
    pred = pred.argmax(dim=1).view(-1)
    true = true.view(-1)
    mat = torch.zeros(c, c, dtype=torch.long)
    mat = mat.index_put_((true, pred), torch.tensor(1), accumulate=True)

    return mat.double() / mat.sum()


def cm_grad(pred: torch.tensor, true: torch.tensor):
    """
    Gradient of confusion matrix
    """
    c = pred.shape[1]
    pred = pred.softmax(dim=1).view(-1, c)
    true = true.view(-1)
    mat = torch.zeros(c, c, dtype=pred.dtype).index_add(0, true, pred) 

    return mat.double() / mat.sum()


def iou_torch(pred: torch.tensor, true: torch.tensor) -> float:
    """
    Intersection over Union using Confusion matrix
    PyTorch version
    does not work with the batch, sinlge images only
    """
    c = pred.shape[1]
    # Remove classes from unlabeled pixels in gt image 
    pred = pred * (true > 0)
    pred = pred.view(-1) # flatten predictions
    true = true.view(-1)
    mat = torch.zeros(c, c, dtype=torch.long) # confusion matrix shape
    mat = mat.index_put_((true, pred), torch.tensor(1), accumulate=True)

    return mat.diag() / (mat.sum(0) + mat.sum(1) - mat.diag()).clamp(1e-8)


def precision_at(iou: np.array, threshold: float = 0.5) -> tuple:
    """ Get true positive, false positive, false negative at iou threshold
    """
    matches = iou >= threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

if __name__ == "__main__":
    test_iou()

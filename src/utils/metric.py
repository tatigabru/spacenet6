import torch 
import numpy as np
import torch.nn.functional as F


def _ignore_channels(*xs, ignore_channels=None, dim=1):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[dim]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=dim, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def tile_iou(output: torch.tensor, target: torch.tensor, numClass: int = NUM_CLASSES):
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
    output = np.asarray(output)
    target = np.asarray(target)
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


def precision_at(iou: np.array, threshold: float = 0.5, iou):
    """ Get true positive, false positive, false negative at iou threshold
    """
    matches = iou >= threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def polygons_iou():


    return iou

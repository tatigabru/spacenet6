"""
Following is my implementation of target LB metric. 
It provides almost exactly same result as visualizer tool, giving a bit worse F1 score than Java implementation, 
but allows to compute it during training after each epoch since it's very fast. Use it at own risk.

__author__: Eugene Khvedchenya
"""
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional


SMOOTH = 1e-6

def bbox2(img: np.array) -> tuple:
    if not img.any():
        return 0, 0, 0, 0
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def bboxes_has_intersection(bbox1, bbox2):
    rmin1, rmax1, cmin1, cmax1 = bbox1
    rmin2, rmax2, cmin2, cmax2 = bbox2

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(cmin1, cmin2)
    yA = max(rmin1, rmin2)
    xB = min(cmax1, cmax2)
    yB = min(rmax1, rmax2)

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return False

    return True


def binary_iou(y_pred, y_true):
    intersection = float((y_pred & y_true).sum())
    union =float( (y_pred | y_true).sum())   
    
    return intersection / (union + 1e-7)


def remove_small_objects(mask: np.ndarray, min_size: Optional[int] = 80):
    """Remove objects smaller then threshold"""


    return mask   


@torch.no_grad()
def buildings_f1_fast(
    pred_mask: np.ndarray, true_mask: np.ndarray, iou_threshold=0.5, min_size: Optional[int] = 80, device: str="cuda"
) -> Tuple[int, int, int]:
    """
    Computes number of TP, FP, FN detections
    :param pred_mask: Instance mask of predicted objects
    :param true_mask: Instance mask of true objects
    :return: tp, fp, fn
    """
    assert pred_mask.shape == true_mask.shape    

    tp = 0
    fp = 0

    #if min_size is not None:
    #    true_mask = remove_small_objects(true_mask, min_size=min_size)
    #    true_mask, _, _ = relabel_sequential(true_mask)
    #    pred_mask = remove_small_objects(pred_mask, min_size=min_size)
    #    pred_mask, _, _ = relabel_sequential(pred_mask)

    num_true_buildings = true_mask.max()
    num_pred_buildings = pred_mask.max()

    true_bboxes = (
        [bbox2(true_mask == label) for label in range(1, num_true_buildings + 1)] if num_true_buildings else []
    )
    pred_bboxes = (
        [bbox2(pred_mask == label) for label in range(1, num_pred_buildings + 1)] if num_pred_buildings else []
    )

    true_mask = torch.from_numpy(true_mask.astype(int)).long().to(device)
    pred_mask = torch.from_numpy(pred_mask.astype(int)).long().to(device)

    found_match = np.zeros(num_true_buildings, dtype=np.bool)

    def compute_iou(x, y):
        i = (x & y).sum()
        u = (x | y).sum()
        return i.item() / (u.item() + 1e-7)

    for pred_building_index in range(num_pred_buildings):

        best_iou = 0
        matching_polygon_index = None

        pred_mask_i = None

        for true_building_index in range(num_true_buildings):
            if found_match[true_building_index]:
                # Skip this truth polygon if it was already matched with another solution polygon.
                continue

            if not bboxes_has_intersection(pred_bboxes[pred_building_index], true_bboxes[true_building_index]):
                continue

            if pred_mask_i is None:
                # Lazy compute mask
                pred_mask_i = pred_mask == pred_building_index + 1

            iou = compute_iou(true_mask == true_building_index + 1, pred_mask_i)

            if iou > best_iou and iou >= iou_threshold:
                # Note the truth polygon which has the highest IOU score if this score is higher than or equal to 0.5.
                # Call this the â€˜matchingâ€™ polygon.
                best_iou = iou
                matching_polygon_index = true_building_index

        if matching_polygon_index is not None:
            # If there is a matching polygon found above, increase the count of true positives by one (TP).
            tp += 1
            found_match[matching_polygon_index] = True

        else:
            # If there is no matching polygon found, increase the count of false positives by one (FP).
            fp += 1

    # When all solution polygons are processed then for each truth polygon that is left unmatched,
    # increase the count of false negatives by one (FN).
    fn = (~found_match).sum()

    return tp, fp, fn
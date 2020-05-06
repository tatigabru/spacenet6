import torch
import torch.nn as nn
import torch.nn.functional as F


def _ignore_channels(*xs, ignore_channels=None, dim=1):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[dim]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=dim, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


class PixelAccuracy(nn.Module):
    __name__ = "pixel_accuracy"

    def __init__(self, ignore_index=-1):
        """Measure pixel accuracy"""
        super().__init__()
        self.ignore_index = ignore_index

    @torch.no_grad()
    def forward(self, output, target):
        """Calculate pixel accuracy between model prediction and target
        Args:
            output (torch.tensor): model output of shape (B,C,H,W);
                each class in one channel (C == n_classes)
            target (torch.tensor): target tensor of shape (B,H,W);
                class objects encoded by unique values
        """
        mask = (target != self.ignore_index)
        output = torch.argmax(output, dim=1)
        correct = (target == output)
        accuracy = (correct * mask).sum().float() / mask.sum()
        return accuracy


class mIoU(nn.Module):
    __name__ = 'mean_iou'

    def __init__(self, ignore_index=None, non_empty=True):
        """Calculate mean IoU score
        Args:
            non_empty (bool): if True, classes which are not presented on target mask,
                and not predicted by `net` are not included to metric calculation.
        """
        super().__init__()
        self.eps = 1e-5
        self.non_empty = non_empty
        self.ignore_index = ignore_index or -1

    @torch.no_grad()
    def forward(self, output, target):
        n_classes = output.shape[1]

        # prepare output
        output = torch.argmax(output, dim=1)

        # convert target to onehot BHWC
        target = F.one_hot(target, n_classes)
        output = F.one_hot(output, n_classes)

        target, output = _ignore_channels(target, output, ignore_channels=[self.ignore_index], dim=-1)

        # compute metric
        intersection = (output * target).sum(dim=(0, 1, 2)).float()
        union = (output + target).sum(dim=(0, 1, 2)).float() - intersection
        if self.non_empty:
            intersection = intersection[union > 0]
            union = union[union > 0]

        iou = (intersection + self.eps) / (union + self.eps)

        return iou.mean()

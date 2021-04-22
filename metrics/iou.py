import numpy as np
import torch


def iou_np(prediction, target):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def iou(predictions: torch.Tensor, targets: torch.Tensor, smooth=1e-6):
    predictions = torch.argmax(predictions, dim=1)  # from shape [B, N, H, W] => [B, H, W]
    intersection = (predictions & targets).float().sum((1, 2))
    union = (predictions | targets).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    return thresholded

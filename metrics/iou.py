import numpy as np
import torch


def iou_np(prediction, target):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def iou(prediction: torch.Tensor, target: torch.Tensor):
    print(prediction.shape)
    print(target.shape)
    return 0

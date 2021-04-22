import numpy as np

from metrics.confusionmatrix import ConfusionMatrix


class IoU:
    def __init__(self, num_classes, normalized=False, ignore_index=None):
        self.conf_matrix = ConfusionMatrix(num_classes, normalized)
        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError('[IoU]: ignore_index must be an int or iterable')

    def reset(self):
        self.conf_matrix.reset()

    def add(self, predicted, target):
        assert predicted.size(0) == target.size(0), \
            '[IoU]: Different batch sizes for predicted and target'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "[IoU]: Predictions must be of dimension (B, H, W) or (B, N, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "[IoU]: Targets must be of dimension (B, H, W) or (B, N, H, W)"
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        else:
            _, target = target.max(1)
        self.conf_matrix.add(predicted.view(-1), target.view(-1))

    def value(self):
        conf_matrix = self.conf_matrix
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        tp = np.diag(conf_matrix)
        fp = np.sum(conf_matrix, 0) - tp
        fn = np.sum(conf_matrix, 1) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = tp / (tp + fp + fn)
        return iou, np.nanmean(iou)

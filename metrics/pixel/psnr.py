# metrics/pixel/psnr.py

import numpy as np
from main.interface import PairwiseMetric
from main.registry import register_metric

class PSNR(PairwiseMetric):
    """
    Peak Signal-to-Noise Ratio (PSNR).

    This metric expects:
    - img_gt, img_pred: HxWxC float32 arrays in the range [0, 1], same shape.
    - mask: (optional) not used in this basic implementation.
    """
    name = "psnr"

    def __init__(self, data_range: float = 1.0):
        self.data_range = data_range

    def __call__(self, img_gt, img_pred, mask=None) -> float:
        mse = np.mean((img_gt - img_pred) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((self.data_range ** 2) / mse)


# Register this metric with the registry
register_metric("psnr", PSNR)

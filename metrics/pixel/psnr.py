import numpy as np
from metrics_hub.interfaces import PairwiseMetric
from metrics_hub.registry import register_metric

class PSNR(PairwiseMetric):
    name = "psnr"

    def __call__(self, img_gt, img_pred, mask=None):
        mse = np.mean((img_gt - img_pred) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * np.log10(1.0 / mse)

register_metric("psnr", PSNR)

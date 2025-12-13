from skimage.metrics import structural_similarity as ssim
from metrics_hub.interfaces import PairwiseMetric
from metrics_hub.registry import register_metric

class SSIM(PairwiseMetric):
    name = "ssim"

    def __call__(self, img_gt, img_pred, mask=None):
        return ssim(img_gt, img_pred, channel_axis=-1, data_range=1.0)

register_metric("ssim", SSIM)

# metrics/structural/ssim.py

from skimage.metrics import structural_similarity as skimage_ssim

from main.interface import PairwiseMetric
from main.registry import register_metric


class SSIM(PairwiseMetric):
    """
    Structural Similarity Index (SSIM)

    Expected input:
      - img_gt, img_pred: HxWxC float32 in [0, 1], RGB, same shape
      - mask: (optional) not used in this MVP implementation
    """
    name = "ssim"

    def __init__(self, data_range: float = 1.0):
        self.data_range = float(data_range)

    def __call__(self, img_gt, img_pred, mask=None) -> float:
        # MVP: ignore mask for now; we will add masked SSIM later.
        score = skimage_ssim(
            img_gt,
            img_pred,
            channel_axis=-1,
            data_range=self.data_range,
        )
        return float(score)


# side-effect registration
register_metric("ssim", SSIM)

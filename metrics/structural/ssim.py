# metrics/structural/ssim.py

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

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
      
        # convert to grayscale for SSIM computation
        # img_gt = rgb2gray(img_gt)
        # img_pred = rgb2gray(img_pred)
        
        # score = ssim(
        #     img_gt,
        #     img_pred,
        #     data_range=1.0,
        #     gaussian_weights=True,
        #     sigma=1.5,
        #     win_size=11,
        #     use_sample_covariance=False,
        # )
        
        # Rgb version SSIM
        score = ssim(
            img_gt,
            img_pred,
            channel_axis=-1,
            data_range=self.data_range,
        )
        
        return float(score)


# side-effect registration
register_metric("ssim", SSIM)

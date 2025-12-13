# metrics/perceptual/lpips.py

import numpy as np
import torch
import lpips
from typing import Optional

from main.interface import PairwiseMetric
from main.registry import register_metric


class LPIPS(PairwiseMetric):
    """
    LPIPS (Learned Perceptual Image Patch Similarity)
    """
    name = "lpips"

    def __init__(self, net: str = "vgg", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = lpips.LPIPS(net=net).eval().to(self.device)

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"LPIPS expects HxWx3 RGB image, got shape={img.shape}")

        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        x = x * 2.0 - 1.0  # [0,1] -> [-1,1]
        return x

    def __call__(self, img_gt, img_pred, mask=None) -> float:
        t1 = self._to_tensor(img_gt).to(self.device)
        t2 = self._to_tensor(img_pred).to(self.device)

        with torch.no_grad():
            d = self.model(t1, t2)
        return float(d.item())


register_metric("lpips", LPIPS)

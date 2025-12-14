# metrics/face/id.py

from typing import Optional

import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet18

from main.interface import PairwiseMetric
from main.registry import register_metric


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


class ID(PairwiseMetric):
    """
    Identity similarity (MVP baseline).

    IMPORTANT:
    - This MVP uses a generic backbone (ResNet18) as a placeholder.
    - For paper-grade face ID similarity, replace the backbone with ArcFace (recommended).
    - Input expected: HxWx3 float32 in [0,1], RGB.
    - Output: cosine similarity in [-1,1], typically [0,1] for similar faces (higher is better).
    """

    name = "id"

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # MVP backbone: ResNet18 feature extractor (replace with ArcFace later)
        backbone = resnet18(weights=None)  # keep deterministic; no auto-download
        backbone.fc = torch.nn.Identity()
        self.model = backbone.eval().to(self.device)

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # ResNet18 typical normalization
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )

    def _embed(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"ID expects HxWx3 RGB image, got shape={img.shape}")

        x = (img * 255.0).clip(0, 255).astype(np.uint8)
        x = self.preprocess(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(x)
        return feat.squeeze(0)

    def __call__(self, img_gt, img_pred, mask=None) -> float:
        # MVP: ignore mask
        f1 = self._embed(img_gt)
        f2 = self._embed(img_pred)
        sim = _cosine_sim(f1, f2)
        return float(sim.item())


register_metric("id", ID)

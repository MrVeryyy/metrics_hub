# metrics/face/id.py

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from main.interface import PairwiseMetric
from main.registry import register_metric


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))


def _available_providers_prefer_cuda() -> List[str]:
    # Prefer CUDA if available, otherwise CPU.
    avail = set(ort.get_available_providers())
    providers: List[str] = []
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers

def _pad_image(img: np.ndarray, pad_ratio: float = 0.25) -> np.ndarray:
    """
    Pad image on all sides by pad_ratio (relative to min(H, W)).
    """
    h, w = img.shape[:2]
    pad = int(min(h, w) * pad_ratio)
    return np.pad(
        img,
        ((pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=0,  # black padding
    )

class ID(PairwiseMetric):
    """
    Face Identity similarity via InsightFace (ArcFace embedding cosine similarity).

    Input:
      - img_gt / img_pred: HxWx3 float32 in [0,1], RGB
    Output:
      - cosine similarity (higher is better; same identity typically close to 1)

    Notes:
      - Uses FaceAnalysis (detection + alignment + recognition).
      - Will pick the largest detected face if multiple faces exist.
    """

    name = "id"

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None,
    ):
        if providers is None:
            providers = _available_providers_prefer_cuda()

        self.providers = providers
        self.device = "cuda" if providers and providers[0] == "CUDAExecutionProvider" else "cpu"
        self.app = FaceAnalysis(name=model_name, providers=providers)

        # ctx_id: 0 for GPU, -1 for CPU. We can set 0 if CUDA provider is present.
        ctx_id = 0 if providers and providers[0] == "CUDAExecutionProvider" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    @staticmethod
    def _rgb01_to_bgr_u8(img: np.ndarray) -> np.ndarray:
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {img.shape}")
        x = (img * 255.0).clip(0, 255).astype(np.uint8)  # RGB uint8
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _pick_largest_face(faces):
        def area(f):
            x1, y1, x2, y2 = f.bbox.astype(np.float32)
            return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        return max(faces, key=area)

    def _embedding(self, img: np.ndarray) -> np.ndarray:
        # add padding to improve face detection on cutout images
        img_pad = _pad_image(img, pad_ratio=0.25)
        
        bgr = self._rgb01_to_bgr_u8(img_pad)
        faces = self.app.get(bgr)
        
        if not faces:
            raise RuntimeError(
                "ID(metric): no face detected. "
                "If your images are face cutouts with tight crop / transparent background, "
                "try adding padding around the face or increasing det_size."
            )
        f = self._pick_largest_face(faces)

        # Prefer normed_embedding (already normalized)
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = f.embedding
        emb = np.asarray(emb, dtype=np.float32)
        return emb

    def __call__(self, img_gt, img_pred, mask=None) -> float:
        e1 = self._embedding(img_gt)
        e2 = self._embedding(img_pred)
        return _cosine(e1, e2)


register_metric("id", ID)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np

from main.io import load_image
from main.registry import get_metric

from pathlib import Path
from .registry import get_metric
from .io import load_image
from .report import save_report

def pairwise(gt_dir, pred_dir, metrics, out_dir="outputs"):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    metric_objs = {m: get_metric(m) for m in metrics}
    results = []

    for gt_path in sorted(gt_dir.iterdir()):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue

        img_gt = load_image(gt_path)
        img_pred = load_image(pred_path)

        row = {"image": gt_path.name}
        for name, metric in metric_objs.items():
            row[name] = float(metric(img_gt, img_pred))
        results.append(row)

    return save_report(results, out_dir)

PathLike = Union[str, Path]

def evaluate_pair(
    img_a: Union[PathLike, np.ndarray],
    img_b: Union[PathLike, np.ndarray],
    metrics: Optional[Iterable[str]] = None,
    lpips_net: str = "vgg",
) -> Dict[str, float]:
    """
    Evaluate pairwise similarity metrics between two images.

    Inputs:
      - img_a, img_b: either file paths (str/Path) or numpy arrays (HxWxC, float32 in [0,1])
      - metrics: iterable of metric names; default is ('psnr','ssim','lpips')
      - lpips_net: LPIPS backbone ('vgg','alex','squeeze') if LPIPS is requested

    Output:
      - dict: {metric_name: score}
        Note: PSNR/SSIM higher is better; LPIPS lower is better.
    """
    if metrics is None:
        metrics = ("psnr", "ssim", "lpips")

    # Load images if paths are provided
    if not isinstance(img_a, np.ndarray):
        img_a = load_image(Path(img_a))
    if not isinstance(img_b, np.ndarray):
        img_b = load_image(Path(img_b))

    # Basic sanity checks
    if img_a.shape != img_b.shape:
        raise ValueError(f"Image shape mismatch: {img_a.shape} vs {img_b.shape}")

    results: Dict[str, float] = {}

    for m in metrics:
        m_lower = m.lower()

        # LPIPS commonly needs configuration (net); others don't
        if m_lower == "lpips":
            metric = get_metric("lpips", net=lpips_net)
        else:
            metric = get_metric(m_lower)

        results[m_lower] = float(metric(img_a, img_b))

    return results

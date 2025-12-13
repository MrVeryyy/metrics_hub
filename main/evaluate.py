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

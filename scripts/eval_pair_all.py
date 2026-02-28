import argparse
from pathlib import Path
import csv
from datetime import datetime
from main.evaluate import evaluate_pair

# Supported image extensions (case-insensitive)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> dict[str, Path]:
    """
    Return mapping: {stem -> file_path} for all supported images under `folder`.
    If multiple files share the same stem, the first encountered is kept.
    """
    mapping: dict[str, Path] = {}
    if not folder.exists():
        return mapping

    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            stem = p.stem
            # keep first occurrence to avoid non-deterministic overwrite
            mapping.setdefault(stem, p)
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate pairwise image similarity metrics from data_pair/A and data_pair/B"
    )

    # Default paths: metrics_hub/data_pair/A & metrics_hub/data_pair/B
    # This script is expected to be run from repo root, but we resolve relative paths robustly.
    parser.add_argument(
        "--a_dir",
        default="data_pair/A",
        help="Folder containing A-side images (default: data_pair/A)",
    )
    parser.add_argument(
        "--b_dir",
        default="data_pair/B",
        help="Folder containing B-side images (default: data_pair/B)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips", "id"],
        help="Metrics to compute (default: psnr ssim lpips id)",
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--dig",
        type=int,
        default=None,
        help="Decimal digits for output rounding (default: no rounding)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any pair is missing A or B side (default: warn and skip)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent  # metrics_hub/
    a_dir = (repo_root / args.a_dir).resolve()
    b_dir = (repo_root / args.b_dir).resolve()

    data_pair_dir = repo_root / "data_pair"

    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    a_map = list_images(a_dir)
    b_map = list_images(b_dir)

    if not a_map:
        raise FileNotFoundError(f"No images found in A folder: {a_dir}")
    if not b_map:
        raise FileNotFoundError(f"No images found in B folder: {b_dir}")

    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()))
    only_a = sorted(set(a_map.keys()) - set(b_map.keys()))
    only_b = sorted(set(b_map.keys()) - set(a_map.keys()))

    if only_a or only_b:
        msg_lines = []
        if only_a:
            msg_lines.append(f"Missing in B ({len(only_a)}): {only_a[:10]}{' ...' if len(only_a) > 10 else ''}")
        if only_b:
            msg_lines.append(f"Missing in A ({len(only_b)}): {only_b[:10]}{' ...' if len(only_b) > 10 else ''}")
        msg = "\n".join(msg_lines)

        if args.strict:
            raise RuntimeError("Pairing mismatch detected (strict mode):\n" + msg)
        else:
            print("WARNING: Pairing mismatch detected; unmatched files will be skipped.")
            print(msg)

    if not common_ids:
        raise RuntimeError(f"No matching pair_ids found between:\n  A: {a_dir}\n  B: {b_dir}")

    rows = []
    failed = 0

    print(f"Found {len(common_ids)} matched pairs.")
    print(f"Computing metrics: {args.metrics}")

    for pid in common_ids:
        gt_path = a_map[pid]
        pred_path = b_map[pid]

        try:
            results = evaluate_pair(
                str(gt_path),
                str(pred_path),
                metrics=args.metrics,
            )

            # presentation-level rounding only
            if args.dig is not None:
                results = {k: round(v, args.dig) for k, v in results.items()}

            row = {
                "pair_id": pid,
                "gt": gt_path.relative_to(data_pair_dir).as_posix(),
                "pred": pred_path.relative_to(data_pair_dir).as_posix(),
                **results,
            }
            rows.append(row)

        except Exception as e:
            failed += 1
            print(f"[FAIL] pair_id={pid}  gt={gt_path.name}  pred={pred_path.name}  err={e}")

    # Prepare CSV output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{timestamp}_result.csv"
    csv_path = out_dir / csv_name

    # CSV columns: fixed + metrics (preserve user order)
    fieldnames = ["pair_id", "gt", "pred"] + list(args.metrics)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # ensure all metric columns exist (in case evaluate_pair omitted something)
            for m in args.metrics:
                r.setdefault(m, "")
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print("\n=== Batch Evaluation Summary ===")
    print(f"Matched pairs: {len(common_ids)}")
    print(f"Succeeded:     {len(rows)}")
    print(f"Failed:        {failed}")
    print(f"Saved CSV to:  {csv_path}")

    # Optional: print quick aggregate (mean) to console (not saved)
    # Keeping it minimal here; you can compute means later from CSV.


if __name__ == "__main__":
    main()

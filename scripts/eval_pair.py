# scripts/eval_pair.py

import argparse
from pathlib import Path
import json
from main.evaluate import evaluate_pair

def main():
    parser = argparse.ArgumentParser(description="Evaluate pairwise image similarity metrics")
    parser.add_argument("--gt", required=True, help="Path to ground-truth image")
    parser.add_argument("--pred", required=True, help="Path to predicted/reconstructed image")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--dig",
        type=int,
        default=None,
        help="Decimal digits for output rounding (default: no rounding)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_pair(
        args.gt,
        args.pred,
        metrics=args.metrics,
    )

    # presentation-level formatting only
    if args.dig is not None:
        results = {k: round(v, args.dig) for k, v in results.items()}

    print("=== Pairwise Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "gt": args.gt,
                "pred": args.pred,
                "metrics": results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()

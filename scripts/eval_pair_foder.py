import argparse
from main.evaluate import pairwise

parser = argparse.ArgumentParser()
parser.add_argument("--gt", required=True)
parser.add_argument("--pred", required=True)
parser.add_argument("--metrics", nargs="+", default=["psnr","ssim","lpips"])
parser.add_argument("--out", default="outputs")
args = parser.parse_args()

pairwise(args.gt, args.pred, args.metrics, args.out)

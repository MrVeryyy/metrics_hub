import lpips
import torch
from main.interface import PairwiseMetric
from main.registry import register_metric

class LPIPS(PairwiseMetric):
    name = "lpips"

    def __init__(self, net="vgg"):
        self.model = lpips.LPIPS(net=net).eval()

    def __call__(self, img_gt, img_pred, mask=None):
        t1 = torch.from_numpy(img_gt).permute(2,0,1).unsqueeze(0)
        t2 = torch.from_numpy(img_pred).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(t1, t2).item())

register_metric("lpips", LPIPS)

from abc import ABC, abstractmethod

class PairwiseMetric(ABC):
    name: str

    @abstractmethod
    def __call__(self, img_gt, img_pred, mask=None) -> float:
        pass


class DistributionMetric(ABC):
    name: str

    @abstractmethod
    def compute(self, real_dir: str, fake_dir: str) -> float:
        pass

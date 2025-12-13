# main/__init__.py

# Trigger metric registration (side effects)
from metrics.pixel import psnr       # noqa: F401
from metrics.structural import ssim  # noqa: F401
from metrics.perceptual import lpips # noqa: F401

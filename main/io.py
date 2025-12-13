import cv2
import numpy as np

def load_image(path, to_rgb=True, normalize=True):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    if normalize:
        img /= 255.0
    return img

def load_mask(path, threshold=0.5):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    return (mask > threshold).astype(np.float32)

# Metrics_Hub

**Metrics_Hub** is a unified, reproducible evaluation toolkit for **image similarity**, **perceptual quality**, and **generative model evaluation**.  
It provides standardized implementations of widely used metrics in computer vision and generative modeling, with **mask-aware evaluation** support for foreground/background analysis.

This repository is designed for **research**, **benchmarking**, and **paper-ready reporting**.

---

## ✨ Key Features

- Unified API for pairwise, masked, and distribution-based evaluation  
- Research-standard metrics used in CVPR / ICCV / ECCV / NeurIPS papers  
- Mask-aware evaluation (foreground / background / custom masks)  
- Reproducible outputs (CSV / JSON / Markdown tables)  
- Paper- and PPT-ready result formatting  
- Modular, extensible design  

---

## 📊 Supported Metrics

### Pixel-wise Metrics
- MSE
- MAE
- PSNR

### Structural / Perceptual Metrics
- SSIM
- MS-SSIM
- GMSD
- FSIM

### Deep Perceptual Metrics
- LPIPS (AlexNet / VGG / SqueezeNet)
- DISTS

### Generative Model Metrics
- FID
- KID
- Inception Score

### Face-specific Metrics
- Identity Similarity (ArcFace / CosFace)

---

## 🧠 Typical Use Cases

- GAN inversion and reconstruction evaluation  
- Image-to-image translation benchmarking  
- Background consistency analysis  
- Face identity preservation evaluation  
- Ablation studies and quantitative comparisons  
- Paper and supplementary material preparation  

---

## 🚀 Installation

```bash
conda env create -f environment.yml
conda activate metrics_hub

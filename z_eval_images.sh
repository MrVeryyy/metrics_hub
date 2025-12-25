#!/bin/bash

# black
# img1="data/image2/083_aligned_bg.jpg"
# img2="data/image2/pti_083_aligned_bg.png"

# white
img1="data/image2/083_aligned_bg.jpg"
img2="data/mean/083_aligned_bg_gray_iter1_MaskPoseIn_Foreground_black.jpg"

dig=3


env=$HOME"/miniconda3/envs/metrics_hub/bin/python"
${env} scripts/eval_pair.py \
   --gt ${img1} \
   --pred ${img2} \
   --dig ${dig}


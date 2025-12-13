#!/bin/bash

img1="data/image1/gt.jpg"
img2="data/image1/psp.jpg"

dig=3

python scripts/eval_pair.py \
   --gt ${img1} \
   --pred ${img2} \
   --dig ${dig}
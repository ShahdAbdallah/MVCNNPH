#!/bin/bash
mkdir -p ../logs

python3 train_multiview_early_fusion_mse_aug_oversample.py \
  --target_len 150 \
  --epochs 500 \
  > ../logs/multiview_early_fusion_mse_aug_oversample.log 2>&1


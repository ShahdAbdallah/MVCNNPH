#!/bin/bash
mkdir -p ../logs
python3 train_multiview_early_fusion_camera_check_len120.py \
  --target_exercise E0 \
  --target_len 120 \
  --epochs 500 \
  --batch_size 256 \
  > ../logs/multiview_E0_early_fusion_camera_check_len120.log 2>&1

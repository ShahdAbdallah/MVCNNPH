#!/bin/bash

mkdir -p ../logs

python3 train_multiview_E3_early_fusion_velocity_only.py \
  --target_len 150 \
  --epochs 500 \
  > ../logs/multiview_E3_early_fusion_velocity_only.log 2>&1

python3 train_multiview_E3_late_fusion_velocity_only.py \
  --target_len 150 \
  --epochs 500 \
  > ../logs/multiview_E3_late_fusion_velocity_only.log 2>&1

python3 train_multiview_E3_middle_fusion_velocity_only.py \
  --target_len 150 \
  --epochs 500 \
  > ../logs/multiview_E3_middle_fusion_velocity_only.log 2>&1

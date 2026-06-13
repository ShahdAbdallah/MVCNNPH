#!/bin/bash

mkdir -p ../logs

python3 train_multiview_middle_fusion_all_except_E3_E7_E9_len120.py \
  --target_len 120 \
  --epochs 500 \
  --batch_size 256 \
  > ../logs/multiview_middle_fusion_all_except_E3_E7_E9_len120.log 2>&1

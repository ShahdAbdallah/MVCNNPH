#!/bin/bash

mkdir -p ../logs

python3 train_multiview_early_fusion_velocity_embedding_weighted.py \
  --target_len 150 \
  --epochs 500 \
  > ../logs/multiview_early_fusion.log 2>&1

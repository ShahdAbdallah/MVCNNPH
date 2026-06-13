#!/bin/bash

mkdir -p ../logs

python3 train_single_E0_C1_C2_velocity_only.py \
  --target_len 150 \
  --epochs 500 \
  > ../logs/single_E0_C1_C2_velocity_only.log 2>&1

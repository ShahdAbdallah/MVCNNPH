#!/bin/bash

mkdir -p ../logs

python3 train_single_E3_all_cameras_velocity_only.py \
  --target_len 150 \
  --epochs 500 \
  > ../logs/single_E3_all_cameras_velocity_only.log 2>&1

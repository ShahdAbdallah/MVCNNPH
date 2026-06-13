#!/bin/bash
mkdir -p ../logs
python3 train_single_exercise_camera_check_len120.py \
  --target_exercise E1 \
  --target_len 120 \
  --epochs 500 \
  --batch_size 256 \
  > ../logs/single_E1_camera_check_len120.log 2>&1

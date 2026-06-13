#!/bin/bash

mkdir -p ../logs

python3 train_single_E3_random_split_velocity_acceleration_bs16.py \
  --target_len 150 \
  --epochs 500 \
  --batch_size 16 \
  > ../logs/single_E3_random_split_velocity_acceleration_bs16.log 2>&1

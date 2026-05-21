#!/bin/bash

mkdir -p ../logs

python3 train_person_exp3_mse_oversample_aug.py \
  --target_len 150 \
  --epochs 500 \
  > ../logs/exp3_mse_oversample_aug.log 2>&1

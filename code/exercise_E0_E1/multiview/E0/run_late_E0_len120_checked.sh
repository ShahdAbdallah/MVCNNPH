#!/bin/bash
set -e

python3 train_multiview_late_fusion_len120_train_val_test_checked.py --exercise E0 --target_len 120 --epochs 500 --batch_name batch_train_val_test_checked_v2

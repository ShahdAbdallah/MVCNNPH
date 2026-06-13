#!/bin/bash

python3 train_exp2_finetune.py --camera C0 --target_len 150 --epochs 300

python3 train_exp2_finetune.py --camera C1 --target_len 150 --epochs 300

python3 train_exp2_finetune.py --camera C2 --target_len 150 --epochs 300

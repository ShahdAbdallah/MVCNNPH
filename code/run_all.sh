#!/bin/bash

python3 train_single_view.py --camera C0 --mode exp2 --target_len 150
python3 train_single_view.py --camera C0 --mode exp3 --target_len 150

python3 train_single_view.py --camera C1 --mode exp2 --target_len 150
python3 train_single_view.py --camera C1 --mode exp3 --target_len 150

python3 train_single_view.py --camera C2 --mode exp2 --target_len 100
python3 train_single_view.py --camera C2 --mode exp3 --target_len 150

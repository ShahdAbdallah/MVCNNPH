#!/bin/bash

echo "Running TCN experiments..."

# C0
python3 train_tcn_single_view.py --camera C0 --target_len 150

# C1
python3 train_tcn_single_view.py --camera C1 --target_len 150

# C2
python3 train_tcn_single_view.py --camera C2 --target_len 150

echo "Done!"


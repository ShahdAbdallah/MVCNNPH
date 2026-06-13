#!/bin/bash
set -e

cd /mvdlph/shahd/MVCNNPH/code

python3 train_multiview_E0_middle_fusion_velocity_only_modified.py \
  --target_len 120 \
  --epochs 500

echo "Middle Fusion E1 results:"
echo "/mvdlph/shahd/MVCNNPH/results_multiview_E1_middle_fusion_velocity_only_modified/len120"

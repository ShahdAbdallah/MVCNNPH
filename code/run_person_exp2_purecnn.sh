#!/bin/bash

OUTPUT_LOG="training_output_allviews.txt"

echo "===================================" | tee $OUTPUT_LOG
echo "Starting training..." | tee -a $OUTPUT_LOG
echo "===================================" | tee -a $OUTPUT_LOG

python3 train_person_exp2_purecnn.py \
    --target_len 150 \
    --epochs 500 2>&1 | tee -a $OUTPUT_LOG

echo "" | tee -a $OUTPUT_LOG
echo "===================================" | tee -a $OUTPUT_LOG
echo "Training finished." | tee -a $OUTPUT_LOG
echo "===================================" | tee -a $OUTPUT_LOG

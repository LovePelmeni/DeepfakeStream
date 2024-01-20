#!/bin/bash 

# Script for automatic training run
# All you have to do is specify configuration in 'pipeline.env' 
# or expose those variables manually inside the terminal session.

# Usage Guide
# 1. run source <path-to-env-file> in order to export configuration. 
# Make sure you've set up all variables, specified in the bellow command.
# 2. run this shell script using 'sh train_pipeline.sh'.s

python3 -u -m src.pipelines.train_pipeline.py \
 --train-data-path $TRAIN_DATA_PATH \
 --val-data-path $VAL_DATA_PATH \
 --labels-path $LABELS_PATH \
 --output-path $OUTPUT_PATH \
 --config-path $CONFIG_PATH \
 --checkpoint-path $CHECKPOINT_PATH \
 --log-path $LOG_PATH \
 --use-cuda $ENABLE_CUDA \
 --use-cpu $ENABLE_CPU \
 --num-workers $NUM_CPU_WORKERS \
 --gpu-id $GPU_ID \
 --use-cudnn-bench $ENABLE_CUDNN_BENCH \
 --seed $SEED \
 --prefix $EXP_PREFIX \
 --log-execution $ENABLE_LOGGING



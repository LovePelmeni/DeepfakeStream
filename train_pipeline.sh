#!/bin/sh

# Script for automatic training run
# All you have to do is specify configuration in 'pipeline.env' 
# or expose those variables manually inside the terminal session.

# Usage Guide
# 1. run source <path-to-env-file> in order to export configuration. 
# Make sure you've set up all variables, specified in the bellow command.
# 2. run this shell script using 'sh train_pipeline.sh'.

# NOTE:
#   you need to use one of the flags:
#       - --use-cuda
#       - --use-mps
#       - --use-cpu
# Do not try to use them at the same time as multiple backends usage is not supported.

FILE=./env_vars/train_pipeline.env

if [ -f "$FILE" ];
then source $FILE;
else
    echo "$FILE does not exist.\
    Training pipeline loads configuration from this file.\
    Create this file and provide settings, according to 'docs/data_management/TRAIN_PIPELINE.md'"
fi

python3 -u -m src.pipelines.train_pipeline \
 --train-data-path $TRAIN_DATA_PATH \
 --train-labels-path $TRAIN_LABELS_PATH \
 --val-data-path $VAL_DATA_PATH \
 --val-labels-path $VAL_LABELS_PATH \
 --output-path $OUTPUT_PATH \
 --config-path $CONFIG_PATH \
 --checkpoint-path $CHECKPOINT_PATH \
 --log-path $LOG_PATH \
 --use-cpu $ENABLE_CPU \
 --num-workers $NUM_CPU_WORKERS \
 --gpu-id $GPU_ID \
 --use-cudnn-bench $ENABLE_CUDNN_BENCH \
 --seed $SEED \
 --log-execution $ENABLE_LOGGING \


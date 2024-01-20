#!/bin/bash 

# Script for automatic training run
# All you have to do is specify configuration in 'pipeline.env' 
# or expose those variables manually inside the terminal session

python training/pipelines/train_classifier.py \
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


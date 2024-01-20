#!/bin/bash 

# Shell script
# for automating data preprocessing

# Usage guide
# 1. set up environment variables, listed down below.
# 2. DATA_CONFIG_DIR - full path to your data configuration .json file
# 3. DATA_DIR - full path to your dataset.
# 4. AUGMENTED_DATA_DIR - full path, where your augmented data is going to be stored after preprocessing

python3 -u -m src.pipelines.preproc_pipeline \
 --config-dir $DATA_CONFIG_DIR \
 --data-dir $DATA_DIR \
 --output-dir $AUGMENTED_DATA_DIR





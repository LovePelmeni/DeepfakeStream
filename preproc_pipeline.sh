#!/bin/sh

# Shell script
# for automating data preprocessing

# Usage guide
# 1. set up environment variables, listed down below.
# 2. DATA_CONFIG_DIR - full path to your data configuration .json file
# 3. DATA_DIR - full path to your dataset.
# 4. AUGMENTED_DATA_DIR - full path, where your augmented data is going to be stored after preprocessing

MODE=$1

if [ "$MODE" == "train" ];
then 
    FILE=./env_vars/train_preproc_pipeline.env
    if [ -f "$FILE" ]; 
    then source $FILE
    else echo "$FILE does not exist.\n
    Shell script loads configuration for training dataset from this file.\n
    Before usage, you need to create this file and set up configuration for your training data.\n
    For more details, see 'docs/data_management/DATA_PIPELINE.md'."
    fi;
elif [ "$MODE" == "validation" ];
then 
    FILE=./env_vars/val_preproc_pipeline.env
    
    if [ -f "$FILE" ]; 
    then source $FILE
    else echo "$FILE does not exist.\n
    Shell script loads configuration for validation dataset from this file.\n
    Before usage, you need to create this file and set up configuration for your validation data.\n
    For more details, see 'docs/data_management/DATA_PIPELINE.md'."
    fi;
else 
    echo "Invalid MODE parameter provided: should be either 'train' or 'validation'. \n
    'train' means, it will load .env configuration for the training dataset, \n
    provided under 'env_vars/train_data_pipeline.env' path. 'validation' means, it will load .env configuration for validation dataset \
    provided under 'env_vars/val_data_pipeline.env' path. You can use only these options, nothing else. \n
    For more details check 'docs/data_management/DATA_PIPELINE.md'."
    exit 1;
fi

python3 -u -m src.pipelines.preproc_pipeline \
 --data-config-dir $DATA_CONFIG_DIR \
 --orig-data-dir $ORIG_DATA_DIR \
 --fake-data-dir $FAKE_DATA_DIR \
 --orig-crop-dir $ORIG_OUTPUT_DIR \
 --fake-crop-dir $FAKE_OUTPUT_DIR \
 --dataset-type "$MODE"


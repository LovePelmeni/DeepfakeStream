# Technical Guide for CLI-based Data Preprocesing Pipeline

This document covers some technical aspects you
need to know about in order to use data preprocessing pipeline, used in the project.

# Assumptions

We want to be able to experiment 
with different data transformations in an efficient way, 
that would allow us to iterate quickly

# Technical Guide 

The source code for the CLI is available ["here"](https://github.com/LovePelmeni/DeepfakeStream/blob/main/src/pipelines/train_pipeline.py).


### Key CLI Arguments:

Here is the list of arguments, supported by the CLI 
and their corresponding descriptions

#### Image Data Configuration

- `config_dir` - path to .json file, containing configuration of the data. It comprises of following properties: 

Json-like file, containing important information about dataset image properties.

-- resize_height - new height to resize images to (used inside augmentations).

-- resize_width - new width to resize images to (used inside augmentations).

-- dataset_type - type of the dataset: 'train', 'valid' or 'test'. It is used for picking right augmentations, based on the type of data.

-- data_type - data precision to use before saving, common choices are float16 and float32

#### Data directory
- `data_dir` - path to the directory, containing your data. NOTE: images should have the same properties, as you've specified inside the configuration, i.e the sizes

#### Output directory
- `output_dir` - this is the path (which may exist or not) where augmented data is going to be stored after pipeline execution.

# How to run CLI

1. go to the main directory of the project.
2. export environment variables or set values manually.
3. run following command in the terminal.

```
python3 -u -m src.pipelines.preproc_pipeline \
 --config-dir $DATA_CONFIG_DIR \
 --data-dir $DATA_DIR \
 --output-dir $AUGMENTED_DATA_DIR
```
4. after pipeline will done executing, you will find your dataset of augmented images under `output_dir` directory.
# Technical Guide for CLI-based Training Pipeline

This document covers some technical aspects you
need to know about in order to use data preprocessing pipeline, used in the project.

# Assumptions

We want to be able to experiment 
with different (models + data) combinations in an efficient way, 
that would allow us to iterate quickly.

# Technical Guide 

The source code for the CLI is available ["here"](https://github.com/LovePelmeni/DeepfakeStream/blob/main/src/pipelines/train_pipeline.py).


### Key CLI Arguments:

Here is the list of arguments, supported by the CLI 
and their corresponding descriptions

#### Image Data Configuration

- `config_dir` - path to .json file, containing configuration of the data. It comprises of following properties: 

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
# Technical Guide for CLI-based Data Preprocesing Pipeline

This document covers some technical aspects you
need to know about in order to use data preprocessing pipeline, used in the project.

# Assumptions

We want to be able to experiment 
with different data transformations in an efficient way, 
that would allow us to iterate quickly.

# Technical Guide 

The source code for the CLI is available ["here"](https://github.com/LovePelmeni/DeepfakeStream/blob/main/src/pipelines/preproc_pipeline.py).


### Key CLI Arguments:

Here is the list of arguments, supported by the CLI 
and their corresponding descriptions

#### Image Data Configuration

`--json-data-config-path` - path to .json file, containing configuration of the data. It contains following properties:

#### Properties

- `mtcnn_image_size` - image size, required by MTCNN face detector. MTCNN requires images to have specific size of `nxn`. The `mtcnn_image_size` parameter is used in augmentations to resize video frames, according to this requirement.

- `encoder_image_size` - size for rescaling cropped face. Example: we extracted face of size (220x180) from the video frame, then it will be rescaled to `encoder_image_size x encoder_image_size` before saving.

- `min_face_size` - minimum size of the face to detect

- `data_type` - ("fp16", "fp32", "int16", "int32") - data precision to use before saving, common choices are float16 and float32

- `frames_per_vid_ratio` - (float) - percentage of frames to extract from the video. Example: 0.01, 0.5, 0.85, etc...

#### Data directory
- `data_dir` - path to the directory, containing your data. NOTE: images should have the same properties, as you've specified inside the configuration, i.e the sizes

#### Output directory
- `--crop_dir` - this is the path (which may exist or not) where augmented data is going to be stored after pipeline execution.

# How to run CLI

##### 1. go to the main directory of the project.
##### 2. export environment variables or set values manually.
##### 3. run following command in the terminal: 

```
python3 -u -m src.pipelines.preproc_pipeline \
 --json-data-config-path $JSON_DATA_CONFIG_DIR \
 --data-dir $DATA_DIR \
 --csv-labels-crop-path $CSV_LABELS_CROP_PATH \
 --crop-dir $OUTPUT_CROP_DIR \
 --dataset-type "$MODE"
```
##### 4. after pipeline done executing, you will find your dataset of augmented images under `--crop-dir` parameter and it's labels under `--json-crops-labels-path` parameter.


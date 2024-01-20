import numpy 
import torch 
import argparse 
import pipelines.train_utils as utils
import pathlib
from src.augmentations import augmentations
import cv2
import os
import logging

"""
Data preprocessing pipeline
for automating processes of cleansing,
transforming and loading data at different scales.

Interaction is available via basic CLI utility.

"""

Logger = logging.getLogger("data_pipeline_logger")
handler = logging.FileHandler(filename="data_pipeline.log")

def data_pipeline():

    parser = argparse.ArgumentParser(description="CLI-based data pipeline")
    arg = parser.add_argument 

    arg("--data-dir", type=str, dest='data_dir', required=True, help='path to dataset directory')
    arg('--data-config', type=str, dest='config_dir', required=True, help='configuration .json file, containing information about the data')
    arg("--output-dir", type=str, dest='output_dir', required=True, help='path where to save augmented data')

    args = parser.parse_args()

    # parsing directories and other data arguments

    config_dir = pathlib.Path(args.config_dir)
    output_dir = pathlib.Path(args.output_dir)
    data_dir = pathlib.Path(args.data_dir)

    output_dir.mkdir(exist_ok=True)

    img_config = utils.load_config(config_path=config_dir)
    dataset_type = img_config['dataset_type']
    data_type = img_config['data_type']

    # picking augmentations

    if dataset_type == "train":

        augmentations = augmentations.get_train_augmentations(
            HEIGHT=img_config['height'],
            WIDTH=img_config['width']
        )

    elif dataset_type == 'valid':

        augmentations = augmentations.get_validation_augmentations(
            HEIGHT=img_config['height'],
            WIDTH=img_config['width']
        )

    # iterating over the entire dataset of images and applying 
    # transformations, based on the type of the dataset

    for img_file_path in os.listdir(data_dir):

        full_path = os.path.join(data_dir.__str__, img_file_path)
        output_img_path = os.path.join(output_dir.__str__, img_file_path)
    
        try:
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED).astype(dtype=data_type)
            augmented_img = augmentations(image=img)['image']

        except(FileNotFoundError) as file_err:
            raise RuntimeError("file not found. \
            sProvided directory may be invalid or does not exist.")
        
        except(Exception) as err:
            Logger.error(err)
            
        try:
            success = cv2.imwrite(filename=output_img_path, img=augmented_img)
            if not success: raise RuntimeError("failed to save image to the provided location")
        except(Exception) as save_err:
            Logger.error(save_err)

data_pipeline()
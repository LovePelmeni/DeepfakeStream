import argparse
import src.pipelines.train_utils as utils
import pathlib
from src.augmentations import augmentations

import cv2
import os
import logging
import sys

"""
Data preprocessing pipeline
for automating processes of cleansing,
transforming and loading data at different scales.

Interaction is available via basic CLI utility.
"""

runtime_logger = logging.getLogger("preproc_pipeline_logger")
err_logger = logging.getLogger(name="preproc_pipeline_err_logger")

runtime_logger.setLevel(level=logging.DEBUG)
err_logger.setLevel(level=logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

runtime_handler = logging.StreamHandler(stream=sys.stdout)
err_handler = logging.FileHandler(filename="preproc_pipeline_error_logs.log")

err_handler.setFormatter(formatter)
runtime_handler.setFormatter(formatter)

runtime_logger.addHandler(runtime_handler)
err_logger.addHandler(err_handler)


def data_pipeline():

    runtime_logger.debug('\n \n1. running preprocessing pipeline... \n')

    parser = argparse.ArgumentParser(
        description="CLI-based Data Preprocessing Pipeline")
    arg = parser.add_argument

    arg("--data-dir", type=str, dest='data_dir',
        required=True, help='path to dataset directory')
    arg('--config-dir', type=str, dest='config_dir', required=True,
        help='configuration .json file, containing information about the data')
    arg("--output-dir", type=str, dest='output_dir',
        required=True, help='path where to save augmented data')

    runtime_logger.debug('2. parsing arguments... \n')
    args = parser.parse_args()

    # parsing directories and other data arguments

    config_dir = pathlib.Path(args.config_dir)
    output_dir = pathlib.Path(args.output_dir)
    data_dir = pathlib.Path(args.data_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    runtime_logger.debug('3. loading configuration files... \n')

    img_config = utils.load_config(config_path=config_dir)

    dataset_type = img_config.get("dataset_type")  # train or valid
    # fp16, int16, int8, etc.
    data_type = utils.resolve_numpy_precision(img_config.get("data_type"))
    img_height = img_config.get("resize_height")  # height of the image
    img_width = img_config.get("resize_width")  # width of the image

    runtime_logger.debug('4. initializing augmentations \n')

    # picking augmentations

    if dataset_type.lower() == "train":

        augments = augmentations.get_training_augmentations(
            HEIGHT=img_height,
            WIDTH=img_width,
        )

    elif dataset_type.lower() == 'valid':

        augments = augmentations.get_validation_augmentations(
            HEIGHT=img_height,
            WIDTH=img_width
        )

    runtime_logger.debug('5. applying transformations... \n \n')

    # iterating over the entire dataset of images and applying
    # transformations, based on the type of the dataset

    for img_file_path in os.listdir(data_dir):

        full_path = os.path.join(data_dir.__str__(), img_file_path)
        output_img_path = os.path.join(output_dir.__str__(), img_file_path)

        try:
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError
            if not img_file_path.endswith("jpeg") and not img_file_path.endswith("jpg"):
                img = utils.convert_to_jpeg(img)
            augmented_img = augments(image=img)['image']

        except (FileNotFoundError):
            raise RuntimeError("file not found. \
            Provided directory may be invalid or does not exist.")

        except (Exception) as err:
            err_logger.error(err)
            raise RuntimeError(err)

        try:
            success = cv2.imwrite(filename=output_img_path,
                                  img=augmented_img.astype(data_type))
            if not success:
                raise RuntimeError(
                    "failed to save image to the provided location")
        except (Exception) as save_err:
            err_logger.error(save_err)

        runtime_logger.debug(
            "PREPROCESSING HAS BEEN SUCCESSFULLY COMPLETED! \n")


if __name__ == '__main__':
    data_pipeline()

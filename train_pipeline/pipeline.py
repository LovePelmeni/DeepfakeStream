from experiments.current_experiment.datasets import datasets
from experiments.current_experiment.image_processing import ImageProcessor
from experiments.current_experiment.augmentations import augmentations

import numpy 
import pandas 
import os
import logging 
import typing

logger = logging.getLogger(__name__)

IMAGE_DATA_PATH = os.environ.get("DATA_PATH")
EMAIL_FOR_ALERT = os.environ.get("EMAIL_FOR_ALERT")
DATA_INFO_PATH = os.environ.get("DATA_INFO_PATH")

class TrainingPipeline(object):
    """
    Pipeline for training ML models
    """
    def load_data(self):
        pass

    def process_images(self, input_imgs: numpy.ndarray):
        pass

    def split_into_datasets(self, imgs: typing.List):
        pass

    def apply_augmentations(self, ):
        return augmentations.apply_augmentations()

    def train_model(self, training_dataset: datasets.DeepFakeClassificationDataset):
        pass 
    
    def evaluate_model(self, validation_dataset: datasets.DeepFakeClassificationDataset):
        pass 

    def send_alert(self, msg: str):
        pass

    def run_steps(self):

        info_dataframe = self.load_data()
        imgs = self.process_images(info_dataframe)

        train_imgs, train_labels, validation_imgs, validation_labels = self.split_into_datasets(imgs)

        aug_train_imgs = self.apply_augmentations(train_imgs)
        aug_valid_imgs = self.apply_augmentations(validation_imgs)

        train_dataset = datasets.get_dataset(aug_train_imgs, train_labels)
        validation_dataset = datasets.get_dataset(aug_valid_imgs, validation_labels)

        cl_loss = self.train_model(train_dataset)
        eval_metric = self.evaluate_model(validation_dataset)

        self.send_alert(
            msg='Successfully trained! \
            Evaluation metric is: %s, \
            Training Loss: %s; ' % (str(eval_metric), str(cl_loss)))

pipeline = TrainingPipeline()
pipeline.run_steps()

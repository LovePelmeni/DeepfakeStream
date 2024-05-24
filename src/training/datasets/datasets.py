from torch.utils import data
import typing
import cv2
import numpy
import torch
import logging

from src.exceptions import exceptions 

logger = logging.getLogger("dataset_logger")


class DeepfakeDataset(data.Dataset):
    """
    Dataset class, used
    for formatting data to make 
    it suitable for training neural network

    Supports several methods:

        - get_numpy_image(idx: int) - (returns simple transformed numpy.ndarray image)

        - get_tensor_image(idx: int) - (
            returns torch.Tensor image, 
            converted in the format, 
            that satifies training needs
        )
        - get_class(idx: int) - returns corresponding label of the image

    Parameters:
    -----------

    input_images - (typing.List) - array of image paths
    input_classes - (typing.List) - array of 

    Methods:
    ---------
        - get_numpy_image - (returns numpy.ndarray object of the image)
        - get_tensor_image - (returns torch.Tensor object of the image)
    """

    def __init__(self,
                 image_paths: typing.List[str],
                 image_labels: typing.List[typing.Union[str, int]],
                 data_type=torch.float32
                 ):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.data_type = data_type

        if any([cv2.imread(img, cv2.IMREAD_UNCHANGED) is None for img in image_paths]):
            raise exceptions.InvalidSourceError(
                msg='some of the image paths provided are not valid'
            )

    def __len__(self):
        return len(self.image_paths)

    def get_numpy_image(self, idx: int) -> numpy.ndarray:

        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_tensor_image(self, idx: int) -> torch.FloatTensor:

        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.get_numpy_image(idx)
        image = torch.from_numpy(image).permute(2, 0, 1).to(self.data_type)
        return image

    def get_class(self, idx: int) -> typing.Union[str, int]:
        return self.image_labels[idx]

    def __getitem__(self, idx: int):
        try:
            image = self.get_tensor_image(idx)
            img_class = self.get_class(idx)
            return image, img_class

        except (Exception) as err:
            logger.error(err)
            raise RuntimeError(
                'failed to parse data, internal error occurred')

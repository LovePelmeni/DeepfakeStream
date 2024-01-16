from torch.utils import data
import typing
import cv2 
import numpy 
import torch
import logging 

logger = logging.getLogger("dataset_logger")

class ImageDataset(data.Dataset):
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
    """
    def __init__(self, 
        image_paths: typing.List[str], 
        image_labels: typing.List[typing.Union[str, int]], 
        augmentations=None
    ):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.images)
    
    def get_numpy_image(self, idx: int) -> numpy.ndarray:
        
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transformations is not None:
            image = self.augmentations(image=image)['image']
        return image
    
    def get_tensor_image(self, idx: int) -> torch.FloatTensor:
        
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transformations is not None:
            image = self.augmentations(image=image)['image']
            
        image = self.get_numpy_image(idx)
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return image
        
    def get_class(self, idx: int) -> typing.Union[str, int]:
        return self.image_labels[idx]

    def __getitem__(self, idx: int):
        try:
            image = self.get_tensor_image(idx)
            img_class = self.get_class(idx)
            return image, img_class
        except(Exception) as err:
            logger.error(err)
            raise RuntimeError(msg='failed to parse data, internal error occurred')





import albumentations
from albumentations import transforms
import cv2
from src.augmentations.resize import IsotropicResize
import numpy

cv2.setNumThreads(0)

def apply_cutout_augmentation(img: numpy.ndarray):
    """
    Applies random cutout augmentation 
    of mouth or half of the face to make network 
    better at considering different features for 
    predicting the final outcome
    
    Args:
        - img - image with human face for augmentation
    """
    pass

def get_training_texture_augmentations(HEIGHT, WIDTH):
    """
    Function returns domain-specific augmentations
    settings for training set
    """
    return albumentations.Compose(
        transforms=[
            transforms.ImageCompression(
                quality_lower=60, 
                quality_upper=100, 
                compression_type=0
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.OneOf(
                transforms=[
                    IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC,
                        target_size=(HEIGHT, WIDTH)
                    ),
                    IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_LINEAR,
                        target_size=(HEIGHT, WIDTH),
                    ),
                    IsotropicResize(
                        interpolation_down=cv2.INTER_LINEAR,
                        interpolation_up=cv2.INTER_LINEAR,
                        target_size=(HEIGHT, WIDTH)
                    ),
                ]
            ),
            albumentations.OneOf(
                transforms=[
                    albumentations.RandomBrightnessContrast(),
                    albumentations.FancyPCA(),
                    albumentations.HueSaturationValue(p=0.5),
                ]
            ),
            albumentations.ToGray(p=0.2),
            albumentations.ShiftScaleRotate(
                shift_limit=1,
                scale_limit=0.7,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT
            )
        ]
    )


def get_validation_augmentations(HEIGHT: int, WIDTH: int):

    return albumentations.Compose(
        transforms=[
            IsotropicResize(
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_LINEAR,
                target_size=(HEIGHT, WIDTH)
            ),
            albumentations.PadIfNeeded(
                min_height=HEIGHT,
                min_width=WIDTH,
                border_mode=cv2.BORDER_CONSTANT
            )
        ]
    )




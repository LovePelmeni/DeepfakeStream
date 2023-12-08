import albumentations
from albumentations import transforms
import cv2
from experiments.current_experiment.augmentations.src import IsotropicResize
import numpy

cv2.setNumThreads(0)

def apply_cutout_augmentation(img: numpy.ndarray):
    pass

def get_training_texture_augmentations():
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
                        interpolation_up=cv2.INTER_CUBIC
                    ),
                    IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_LINEAR,
                    ),
                    IsotropicResize(
                        interpolation_down=cv2.INTER_LINEAR,
                        interpolation_up=cv2.INTER_LINEAR
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


def get_validation_augmentations(size: int):
    return albumentations.Compose(
        transforms=[
            IsotropicResize(
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_LINEAR
            ),
            albumentations.PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT
            )
        ]
    )

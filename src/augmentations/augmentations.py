import albumentations
from albumentations import transforms
import albumentations
import cv2
from src.augmentations import resize
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


def get_training_augmentations(HEIGHT: int, WIDTH: int) -> albumentations.Compose:
    """
    Function returns domain-specific augmentations
    settings for training set

    NOTE:
        expected image need to have .JPEG format.
        Make sure to apply conversion, before
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
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC,
                        target_shape=(HEIGHT, WIDTH),
                    ),
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_LINEAR,
                        target_shape=(HEIGHT, WIDTH),
                    ),
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_LINEAR,
                        interpolation_up=cv2.INTER_LINEAR,
                        target_shape=(HEIGHT, WIDTH)
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
            albumentations.ShiftScaleRotate(
                shift_limit=1,
                scale_limit=0.7,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT
            )
        ]
    )


def get_validation_augmentations(HEIGHT: int, WIDTH: int) -> albumentations.Compose:

    return albumentations.Compose(
        transforms=[
            resize.IsotropicResize(
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_LINEAR,
                target_shape=(HEIGHT, WIDTH)
            ),
            albumentations.PadIfNeeded(
                min_height=HEIGHT,
                min_width=WIDTH,
                border_mode=cv2.BORDER_REFLECT
            )
        ]
    )

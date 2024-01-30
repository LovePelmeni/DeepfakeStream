import albumentations
import cv2
from src.preprocessing import resize

cv2.setNumThreads(0)


def get_training_augmentations(IMAGE_SIZE: int) -> albumentations.Compose:
    """
    Returns augmentations for training data

    NOTE:
        expected image need to have .JPEG format.
        Make sure to apply conversion, before
    """

    return albumentations.Compose(
        transforms=[
            albumentations.OneOf(
                transforms=[
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC,
                        target_size=IMAGE_SIZE,
                    ),
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_LINEAR,
                        target_size=IMAGE_SIZE,
                    ),
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_LINEAR,
                        interpolation_up=cv2.INTER_LINEAR,
                        target_size=IMAGE_SIZE
                    ),
                ]
            ),
            albumentations.ImageCompression(
                quality_lower=60,
                quality_upper=100,
                compression_type=0
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.OneOf(
                transforms=[
                    albumentations.RandomBrightnessContrast(),
                    albumentations.FancyPCA(),
                    albumentations.HueSaturationValue(p=0.5),
                ]
            ),
            albumentations.ShiftScaleRotate(
                shift_limit=1,
                scale_limit=0.5,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT
            )
        ]
    )


def get_validation_augmentations(IMAGE_SIZE: int) -> albumentations.Compose:
    """
    Returns augmentations for training data

    NOTE:
        expected image need to have .JPEG format.
        Make sure to apply conversion, before
    """
    return albumentations.Compose(
        transforms=[
            resize.IsotropicResize(
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_LINEAR,
                target_size=IMAGE_SIZE
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ImageCompression(
                quality_lower=90,
                quality_upper=100,
                compression_type=0,
                p=0.3
            )
        ]
    )

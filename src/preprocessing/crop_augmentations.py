import albumentations
import cv2
from src.preprocessing import resize

cv2.setNumThreads(0)


def get_train_crop_augmentations(CROP_IMAGE_SIZE: int) -> albumentations.Compose:
    """
    Returns augmentations for training data

    NOTE:
        expected image need to have .JPEG format.
        Make sure to apply conversion, before
    """

    return albumentations.Compose(
        transforms=[
            albumentations.ImageCompression(
                quality_lower=60,
                quality_upper=100,
                compression_type=0
            ),
            albumentations.GaussNoise(p=0.1),
            albumentations.GaussianBlur(p=0.05),
            albumentations.RandomGamma(p=0.5),
            albumentations.HorizontalFlip(p=0.5),

            albumentations.OneOf(
                transforms=[
                    albumentations.RandomBrightnessContrast(),
                    albumentations.FancyPCA(),
                    albumentations.HueSaturationValue(),
                ], p=0.7
            ),
            albumentations.OneOf(
                transforms=[
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC,
                        target_size=CROP_IMAGE_SIZE,
                    ),
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_LINEAR,
                        target_size=CROP_IMAGE_SIZE,
                    ),
                    resize.IsotropicResize(
                        interpolation_down=cv2.INTER_LINEAR,
                        interpolation_up=cv2.INTER_LINEAR,
                        target_size=CROP_IMAGE_SIZE
                    ),
                ], p=1),
        ]
    )


def get_validation_crop_augmentations(CROP_IMAGE_SIZE: int) -> albumentations.Compose:
    """
    Returns augmentations for training data

    NOTE:
        expected image need to have .JPEG format.
        Make sure to apply conversion, before
    """
    return albumentations.Compose(
        transforms=[
            albumentations.ImageCompression(
                quality_lower=90,
                quality_upper=100,
                compression_type=0,
                p=0.3
            ),
            resize.IsotropicResize(
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_LINEAR,
                target_size=CROP_IMAGE_SIZE
            ),
            albumentations.HorizontalFlip(p=0.5)
        ]
    )

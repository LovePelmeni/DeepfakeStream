import albumentations
from albumentations import transforms

import albumentations
import cv2

from src.preprocessing import resize
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

def find_dot_intersection(p1, p2):
    return (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

def face_vert_side_blackout(input_img: numpy.ndarray, landmarks: list):
    """
    Function performs blackouts
    side factial region on the face using 
    facial landmarks, generated by the MTCNN model

    NOTE:
        le, re, n, lm, rm - landmark short names from (left eye, right eye, nose, left mouth, right mouth)
        e, n, m - is corresponding parts (eye, nose, mouth) coordinates
        bx, by - coordinates of intersection between m and n coordinates
        ux, uy - coordinates of intersection between e and n coordinates
    """
    le, re, n, lm, rm = landmarks
    e, n, m = numpy.random.choice([(le, n, lm), (re, n, rm)])
    
    bx, by = find_dot_intersection(n, m)
    ux, uy = find_dot_intersection(n, e)

    input_img[e[0]:bx, e[1]:by] = 0
    input_img[m[0]:ux, m[1]:uy] = 0

    return input_img 

def face_horiz_side_blackout(input_img: numpy.ndarray, landmarks: list):
    pass 

def face_nose_blackout(input_img: numpy.ndarray, landmarks: list):
    pass 

def face_eyes_blackout(input_img: numpy.ndarray, landmarks: list):
    pass 

def face_mouth_blackout(input_img: numpy.ndarray, landmarks: list):
    pass 


def get_training_augmentations(HEIGHT: int, WIDTH: int) -> albumentations.Compose:
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
            transforms.ImageCompression(
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


def get_validation_augmentations(HEIGHT: int, WIDTH: int) -> albumentations.Compose:
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
                target_shape=(HEIGHT, WIDTH)
            ),
            transforms.ImageCompression(
                quality_lower=90,
                quality_upper=100,
                compression_type=0,
                p=0.3
            )
        ]
    )


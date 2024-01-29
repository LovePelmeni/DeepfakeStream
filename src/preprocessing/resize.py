from itertools import pairwise
from signal import SIG_DFL
import cv2
import numpy
import albumentations
import typing
import logging 
import sys

Logger = logging.getLogger("aug_logger")
handler = logging.StreamHandler(stream=sys.stdout)
Logger.addHandler(handler)
class IsotropicResize(albumentations.ImageOnlyTransform):

    """
    Class for dynamic isotropic resize of images

    Parameters:
    ----------

    interpolation_down - how to interpolate image when we want to scale it to a smaller size
    interpolation_up - how to interpolate image when we want to scale it to a bigger size
    """

    def __init__(self,
                 target_shape: typing.Tuple[int],
                 interpolation_down=cv2.INTER_AREA,
                 interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False,
                 p=0.5
                 ):
        super(IsotropicResize, self).__init__(always_apply, p)
        
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up
        self.target_shape = target_shape

        if len(self.target_shape) != 2:
            raise RuntimeError(
            "Invalid target image size. Should be tuple of height and width")
    
    def _get_interpolation_policy(self, image, target_size: int):
        """
        Function returns interpolation policy,
        based on the fact how we want to scale our image

        Returns:
            - interpolation cv2 object

        Example:
            image = numpy.eye(1024)
            target_size = 512
            policy = _get_interpolation_policy(image, target_size)

            resized_img = cv2.resize(
                image, 
                (target_size, target_size), 
                interpolation=policy
            )
        """
        height = image.shape[0]
        width = image.shape[1]
        
        if target_size == height and target_size == width:
            return None
        if target_size > height or target_size > width:
            return self.interpolation_up
        else:
            return self.interpolation_down

    @staticmethod
    def get_ratio_size(image: numpy.ndarray, target_size: int):
        """
        Function returns new
        height and width of the image after 
        resizing to the "target size", while preserving it's ratio

        image - numpy.ndarray - image for processing 
        target_size (int) - new size for the transformed image
        """
        if not isinstance(image, numpy.ndarray):
            raise TypeError(
                "image should be a numpy.ndarray, but not %s" % type(image))

        height = image.shape[0]
        width = image.shape[1]

        ratio = width / height

        if ratio > 1:
            new_width = target_size
            new_height = int(new_width / ratio)
        else:
            new_height = target_size
            new_width = int(new_height * ratio)
        return new_height, new_width

    def apply(self, image: numpy.ndarray, **kwargs):
        try:
            if not isinstance(image, numpy.ndarray):
                image = numpy.asarray(image)

            if (image.shape[0] == self.target_shape[0]) and (image.shape[1] == self.target_shape[1]):
                return image

            ratio_h, ratio_w = self.get_ratio_size(image, self.target_shape)
            int_policy = self._get_interpolation_policy(image, self.target_shape)
            
            return cv2.resize(
                image, 
                (ratio_h, ratio_w), 
                interpolation=int_policy
            )

        except (TypeError) as err:
            Logger.error(err)
            raise RuntimeError('Invalid image format, convertion failed')





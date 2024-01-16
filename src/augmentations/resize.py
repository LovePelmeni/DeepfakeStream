import cv2
from PIL.Image import Image
import numpy
import albumentations
import typing


class IsotropicResize(albumentations.ImageOnlyTransform):

    """
    Class for dynamic isotropic resize of images

    Parameters:
    ----------

    interpolation_down - how to interpolate image when we want to scale it to a smaller size
    interpolation_up - how to interpolate image when we want to scale it to a bigger size
    """

    def __init__(self,
                 interpolation_down: cv2.INTER_AREA,
                 interpolation_up: cv2.INTER_CUBIC,
                 target_size: typing.Tuple[int]
                 ):
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up
        self.target_size = target_size

        if len(self.target_size) != 2:
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
        height, width = image.shape[0], len(image[0])
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

        height, width = image.shape[0], image[0].shape[0]
        ratio = width / height

        if ratio > 1:
            new_width = target_size
            new_height = int(new_width / ratio)
        else:
            new_height = target_size
            new_width = int(new_height * ratio)
        return new_height, new_width

    def apply(self, image: numpy.ndarray):
        try:
            if not isinstance(img, numpy.ndarray):
                img = numpy.asarray(img)

            ratio_h, ratio_w = self._get_aspect_ratio(img, self.target_size)
            int_policy = self._get_interpolation_policy(img, self.target_size)
            return cv2.resize(
                img, 
                (ratio_h, ratio_w), 
                interpolation=int_policy
            )

        except (TypeError):
            raise RuntimeError('Invalid image format, convertion failed')






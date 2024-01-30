import cv2
import numpy
import albumentations
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
                 target_size: int,
                 interpolation_down=cv2.INTER_AREA,
                 interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False,
                 p=0.5
                 ):
        super(IsotropicResize, self).__init__(always_apply, p)

        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up
        self.target_size = target_size

    def apply(self, image: numpy.ndarray, **kwargs):

        if not isinstance(image, numpy.ndarray):
            image = numpy.asarray(image)

        height, width = image.shape[:2]

        if max(height, width) == self.target_size:
            return image

        max_height = self.target_size
        max_width = self.target_size

        # Calculate the new dimensions
        new_width = int(width * max_height / height)
        new_height = int(height * max_width / width)

        if new_width > max_width:
            new_width = max_width

        if new_height > max_height:
            new_height = max_height

        scale = max(height / self.target_size, width / self.target_size)
        interpolation = self.interpolation_up if scale > 1 else self.interpolation_down
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation
        )
        return resized

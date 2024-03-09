import unittest 
from src.preprocessing.crop_augmentations import resize
import numpy
import cv2

class IsotropicResizeTestCase(unittest.TestCase):

    def setUp(self):
        self.test_resize_configuration = {
            "image_channels": 3,
            "image_height": 512,
            "image_width": 512,
        }
    
    def test_shape_changed(self):
        img_channels = self.test_resize_configuration.get("image_channels")
        img_height = self.test_resize_configuration.get("image_height")
        img_width = self.test_resize_configuration.get("image_width")
        resizer = resize.IsotropicResize(
            target_size=img_height,
            interpolation_down=cv2.INTER_LINEAR,
            interpolation_up=cv2.INTER_CUBIC
        )
        test_img = numpy.random.rand((img_channels, img_height, img_width))
        resized_img = resizer(image=test_img)['image']
        self.assertIsInstance(
            resized_img, 
            numpy.ndarray, 
            msg='Augmentations should return numpy.ndarray typed object'
        )
        self.assertEqual(resized_img.shape[0], img_height, msg='height does not equal to requested value')
        self.assertEqual(resized_img.shape[1], img_width, msg='width does not equal to requested value')
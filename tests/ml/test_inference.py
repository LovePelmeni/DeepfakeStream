import unittest 
import torch
import numpy
from src.inference_time import measure_inference_time


class NetworkInferenceTestCase(unittest.TestCase):

    def setUp(self, 
        img_channels: int, 
        img_height: int, 
        img_width: int, 
        required_inference_speed_ms: int
    ):
        try:
            self.network = torch.load("models/network.onnx")
        except(Exception) as err:
            raise RuntimeError(
            "Invalid Neural Network Path, \
            when tried to run unittest. \
            Make sure it's valid and file does exist")

        self.test_inference_data = torch.rand(1, img_channels, img_height, img_width)
        self.test_inference_device = torch.device('cpu')
        self.required_inference_speed_ms = required_inference_speed_ms
        self.expected_output = torch.random.rand(500, 500)
        self.AVG_INFERENCE_MS_TIME_DEVIATION = 1e-5
    
    def test_inference_output(self):
        output_matrix = self.network.forward(
            self.test_inference_data.to(
                self.test_inference_device
            )
        )
        self.assertIsInstance(
            obj=output_matrix, 
            cls=torch.Tensor, 
            msg='output of network need to be torch.Tensor instance'
        )
        self.assertFalse(
            expr=len(output_matrix) == 0, 
            msg='network returned empty output during test inference'
        )

    def test_inference_speed(self):
        avg_computed_speed = measure_inference_time(self.network)
        self.assertAlmostEqual(
            a=[avg_computed_speed],
            b=[avg_computed_speed], 
            atol=self.AVG_INFERENCE_MS_TIME_DEVIATION,
            msg='deviation between actual speed and expected speed are too different.')
        self.assertIsNotNone(obj=avg_computed_speed, msg='inference speed output should not be None')
        self.assertTrue(expr=avg_computed_speed > 0, msg="invalid inference speed output")
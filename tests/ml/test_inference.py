import unittest 
import torch

import numpy
from src.inference_time import measure_inference_time

import pytest 
import os

from torch import nn
import cv2


NETWORK_PATH = os.environ.get("NETWORK_PATH")
TEST_INPUT_PATH = os.environ.get("TEST_INPUT_PATH")
REQUIRED_INFERENCE_SPEED_MS = os.environ.get("REQUIRED_INFERENCE_SPEED_MS")

@pytest.fixture(scope='module')
def network() -> nn.Module:
    try:
        return torch.load(f=NETWORK_PATH)
    except(FileNotFoundError, Exception) as err:
        raise RuntimeError("failed to load network, invalid path has been assigned.")

@pytest.fixture(scope='module')
def test_input() -> torch.Tensor:
    return cv2.imread(
        TEST_INPUT_PATH, 
        cv2.IMREAD_UNCHANGED
    )

class NetworkInferenceTestCase(unittest.TestCase):

    def setUp(self):
        self.network = network()
        self.test_inference_data = test_input()
        self.test_inference_device = torch.device('cpu')
        self.required_inference_speed_ms = REQUIRED_INFERENCE_SPEED_MS
        self.expected_output = torch.random.rand(500, 500)
        self.avg_ms_inference_time_deviation = 1e-5
    
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
            atol=self.avg_ms_inference_time_deviation,
            msg='deviation between actual speed and expected speed are too different.')

        self.assertIsNotNone(obj=avg_computed_speed, msg='inference speed output should not be None')
        self.assertTrue(expr=avg_computed_speed > 0, msg="invalid inference speed output")

    def tearDown(self):
        del self.network
        del self.expected_output
        torch.cuda.empty_cache()

class NetworkArithemticTestCase(unittest.TestCase):
    """
    Test case, responsible for checking 
    network's performing correctness.

    Covers some basic health checking properties:
        1. how it overfitts after turning off regularization
        2. verify, that loss decreases after running several training epochs
        3. 
    """
    def setUp(self):
        self.network = network()
        self.test_input = test_input()
        self.test_training_epochs = 5

    def tearDown(self):
        torch.cuda.empty_cache()
        del self.network

    def _turn_off_regularization(self):

        for layer in self.network.layers:

            # disabling droupout layers
            if isinstance(layer, nn.Dropout):
                layer.p = 0.0

            # disabling batch normalization layers
            if isinstance(layer, nn.BatchNorm2d):
                layer.training = False

    def test_network_overfits(self):

        self._turn_off_regularization()
        train_loss = self.train_network(
            epochs=self.test_training_epochs
        )

        self.assertAlmostEqual(
            first=train_loss, second=0.0, 
            msg="loss is not close to zero, \
            as it supposed to be during overfitting: got '%s'" % str(train_loss)
        )

    def test_network_trains_properly(self):

        train_losses = self.train_network(
            epochs=self.test_training_epochs, 
            history=True
        )

        for loss in range(1, len(train_losses)):

            self.assertLess(
                train_losses[loss], 
                train_losses[loss-1], 
                msg='training loss does not decrease during training, as it suppose to'
            )
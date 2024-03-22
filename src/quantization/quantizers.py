from torch import nn
import torch.quantization
import numpy
import torch.ao.quantization
from torch.utils import data
import torch
import typing

class InputQuantizer(object):
    """
    Inference quantizer. Application string
    container string branch application string validator.
    Parameters:
    ----------
        quantization_type - torch.qint8, torch.qint4, etc. Make sure
        you picked the supported quantization type.
        quan_scale - quantization scale
        quan_zero - quantization zero point
    """
    def __init__(self, 
        quantization_type, 
        quan_scale: typing.Union[torch.Tensor, float], 
        quan_zero: typing.Union[torch.Tensor, int]
    ):
        self.quantization_type = quantization_type
        self.quantization_scale = quan_scale
        self.quantization_zero = quan_zero

    def __call__(self, input_img: numpy.ndarray):
        return torch.quantize_per_tensor(
            input_img,
            self.quantization_scale,
            self.quantization_zero,
            self.quantization_type,
        )

class StaticNetworkQuantizer(object):
    """
    Base module for performing static quantization
    of the network.
    """
    def __init__(self, quan_type):
        self.quan_type = quan_type
        self.calibrator = NetworkCalibrator()

    def quantize(self, 
        input_model: nn.Module, 
        calibration_dataset: data.Dataset, 
        calib_batch_size: int
    ):
        calibration_loader = self.calibrator.configure_calibration_loader(
            calibration_dataset=calibration_dataset,
            calibration_batch_size=calib_batch_size,
            loader_workers=2
        )
        # perform calibration
        stat_network = self.calibrator.calibrate(
            input_model,
            loader=calibration_loader,
            q_type=self.quan_type
        )
        quantized_model = torch.quantization.convert(stat_network)
        return quantized_model 

class NetworkCalibrator(object):

    """
    Base module for performing
    calibration for static post-training
    quantization.
    """

    def configure_calibration_loader(self, 
        calibration_dataset: data.Dataset,
        calibration_batch_size: int,
        loader_workers: int = 1
    ):
        """
        Configures data loader for
        performing calibration.
        """
        return data.DataLoader(
            dataset=calibration_dataset,
            batch_size=calibration_batch_size,
            shuffle=True,
            num_workers=loader_workers,
        )

    def calibrate(self, 
        network: nn.Module, 
        loader: data.DataLoader,
        q_type
    ):
        """
        Calibrates given network
        for finding optimal quantization
        parameters.
        
        NOTE:
            loader should contain a dataset,
            which is originally derived from
            the training set.
        """
        network.eval()
        # Specify the quantization configuration
        qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.observer.MinMaxObserver.with_args(dtype=q_type),
            weight=torch.ao.quantization.observer.MinMaxObserver.with_args(dtype=q_type)
        )
        # Apply the quantization configuration to the model
        network.qconfig = qconfig
        stat_network = torch.ao.quantization.prepare(network)

        # performing calibration 
        for images, _ in loader:
            stat_network.forward(images).cpu()

        return stat_network
from torch import nn
import torch.quantization
import numpy
import torch.ao.quantization
from torch.utils import data
import torch
import typing
import logging
from src.quantization import base

quan_logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename='quantization_logs.log')
quan_logger.addHandler(file_handler)

class InputTensorQuantizer(base.BaseInputQuantizer):
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
        q_type,
        pretrained_observer: torch.ao.quantization.observer.ObserverBase
    ):
        self.quantization_type = q_type
        self.observer = pretrained_observer

    def compute_statistics(self, input_image: torch.Tensor):
        scale, zero_point = self.observer.calculate_qparams(input_image)
        return scale, zero_point

    def __call__(self, input_img: numpy.ndarray):
        scale, zero_point = self.compute_statistics(input_image=input_img)
        return torch.quantize_per_tensor(
            input_img,
            scale,
            zero_point,
            self.quantization_type,
        )
class InputChannelQuantizer(base.BaseInputQuantizer):
    """
    Module for quantizing input 
    images per channel.
    
    Parameters:
    -----------
        quan_type - type to use for input quantization. Typically (torch.qint8, torch.qint16).
        input_observer - pretrained observer for computing quantization statistics.
    """
    def __init__(self, 
        quan_type: torch.dtype, 
        input_observer: torch.ao.quantization.observer.ObserverBase
    ):
        self.quan_type = quan_type 
        self.input_observer = input_observer

    def compute_statistics(self, input_image: torch.Tensor):
        return self.input_observer.compute_qparams(input_image)
        
    def quantize(self, input_image: torch.Tensor):
        scales = []
        zero_points = []
        for ch in range(input_image.shape[-1]):
            ch_scale, ch_zero_point = self.compute_statistics(
            input_image[:, :, ch])
            scales.append(ch_scale)
            zero_points.append(ch_zero_point)
        
        return torch.quantize_per_channel(
            input_image,
            scales=scales,
            zero_points=zero_points,
            dtype=self.quan_type,
        )
        

class StaticNetworkQuantizer(object):
    """
    Base module for performing static quantization
    of the network.
    """
    def __init__(self, q_activation_type, q_weight_type):
        self.q_weight_type = q_weight_type 
        self.q_activation_type = q_activation_type 
        self.q_activation_type = q_activation_type
        self.q_weight_type = q_weight_type
        self.calibrator = NetworkCalibrator()

    def quantize(self, 
        input_model: nn.Module, 
        calibration_dataset: data.Dataset, 
        calib_batch_size: int
    ):
        try:
            calibration_loader = self.calibrator.configure_calibration_loader(
                calibration_dataset=calibration_dataset,
                calibration_batch_size=calib_batch_size,
                loader_workers=2
            )
            # perform calibration
            stat_network = self.calibrator.calibrate(
                input_model,
                loader=calibration_loader,
                q_type=self.quan_type,
                weight_q_type=self.q_weight_type,
                activation_q_type=self.q_activation_type
            )
            quantized_model = torch.quantization.convert(stat_network)
            return quantized_model 

        except(Exception) as err:
            quan_logger.error(err)
            return None

class NetworkCalibrator(object):

    """
    Base module for performing
    calibration for static post-training
    quantization.
    """

    def configure_calibration_loader(self, 
        calibration_dataset: data.Dataset,
        calibration_batch_size: int,
        loader_workers: int = 0
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
        activation_q_type: torch.dtype,
        weight_q_type: torch.dtype,
        weight_observer: torch.ao.quantization.observer.ObserverBase,
        activation_observer: torch.ao.quantization.observer.ObserverBase,
    ) -> typing.Union[nn.Module, None]:
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
        try:
            # Specify the quantization configuration
            qconfig = torch.ao.quantization.QConfig(
                activation=activation_observer.with_args(dtype=activation_q_type),
                weight=weight_observer.with_args(dtype=weight_q_type)
            )
            # Apply the quantization configuration to the model
            network.qconfig = qconfig
            stat_network = torch.ao.quantization.prepare(network)

            # performing calibration 
            for images, _ in loader:
                stat_network.forward(images).cpu()

            return stat_network
        except(Exception) as err:
            quan_logger.debug(err)
            return None
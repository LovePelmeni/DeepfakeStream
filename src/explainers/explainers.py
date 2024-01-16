import numpy 
from src.datasets import datasets 
from torch import nn
import typing
import logging 
import matplotlib.pyplot as plt
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import typing

logger = logging.getLogger("explainer_logger")

def interpret_network_predictions(
    network: nn.Module, 
    network_interpretation_layers: typing.List[nn.Module],
    inference_device: torch.device,
    dataset: datasets.ImageDataset,
    img_indices: typing.List
):
        
    cam = GradCAM(
        model=network, 
        target_layers=[network_interpretation_layers]
    )

    _, ax = plt.subplots(ncols=2, nrows=5, figsize=(30, 30))

    for sample_idx in img_indices:
        
        visual_img = dataset.get_numpy_image(sample_idx)
        interpret_img = dataset.get_tensor_image(sample_idx)
        img_class = dataset.get_class(sample_idx)
        
        grayscale_map = cam(
            input_tensor=interpret_img.unsqueeze(0).to(inference_device), 
            targets=[ClassifierOutputTarget(img_class)],
        )[0, :]

        # normalizing image 
        maxR = numpy.max(visual_img.flatten())
        minR = numpy.min(visual_img.flatten())
        norm_img = (visual_img - minR) / (maxR - minR)
        
        visualization = show_cam_on_image(
            norm_img, 
            grayscale_map, 
            use_rgb=True
        )

        ax[sample_idx, 0].imshow(visual_img)
        ax[sample_idx, 1].imshow(visualization)


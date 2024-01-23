import json
import typing
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.adamw import AdamW
from torch.optim.rmsprop import RMSprop
import pandas
import torch
import os
import efficientnet_pytorch as effnet
import logging
import numpy
import cv2
from src.losses import losses
from src.metrics import metrics
from src.datasets import datasets
from src.schedulers import lr_schedulers

Logger = logging.getLogger("utils_logger")


def resolve_torch_precision(precision: typing.Literal['fp16', 'bfp16', 'fp32', 'int8']):
    """
    Returns torch-like precision object
    from the input string
    Parameters:
    -----------
    precision - (str) - precision, represented as a string
    """
    model_dtype = torch.float32

    if precision == 'int8':
        model_dtype = torch.int8

    elif precision == 'fp16':
        model_dtype = torch.float16

    elif precision == 'bfp16':
        model_dtype = torch.bfloat16

    return model_dtype


def resolve_numpy_precision(precision: typing.Literal['fp16', 'fp32', 'int8']):
    """
    Returns numpy-like precision object from 
    input string 
    Parameters:
    -----------
    precision - (str) - precision, represented as a string
    """
    if precision == 'int8':
        return numpy.int8

    elif precision == 'fp16':
        return numpy.float16

    elif precision == 'bfp16':
        return numpy.bfloat16


def load_config(config_path: str) -> typing.Dict:
    with open(config_path, mode='r') as json_config:
        json_file = json.load(json_config)
    json_config.close()
    return json_file


def get_loss_from_config(loss_config: typing.Dict) -> nn.Module:

    loss_name = loss_config.get("name")
    weights = loss_config.get("weights", torch.as_tensor([1, 1]))
    weight_type = resolve_torch_precision(loss_config.get("weight_type", "fp16"))
    label_smoothing_eps = loss_config.get("label_smoothing", 0.0)
     
    if loss_name.lower() == 'focal_loss':

        return losses.FocalLoss(
            weights=weights,
            weight_type=weight_type,
            gamma=loss_config.get("gamma", 2)
        )

    if loss_name.lower() == 'bce_loss' or loss_name.lower() == 'cce_loss':
        return nn.CrossEntropyLoss(
            weight=weights.to(weight_type), 
            label_smoothing=label_smoothing_eps
        )


def get_optimizer(config_dict: typing.Dict, model: nn.Module) -> nn.Module:

    learning_rate = config_dict.get('learning_rate')
    model_params = model.parameters()

    weight_decay = config_dict.get("weight_decay")
    name = config_dict.get("name")

    if name.lower() == 'adam':
        return optim.Adam(
            params=model_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name.lower() == 'sgd':
        momentum = config_dict.get("momentum")
        nesterov = config_dict.get("nesterov", False)
        return optim.SGD(
            params=model_params,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )

    if name.lower() == 'adamw':
        return AdamW(
            params=model_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name.lower() == 'rmsprop':
        momentum = config_dict.get("momentum")
        alpha = config_dict.get("momentum")
        return RMSprop(
            params=model_params,
            lr=learning_rate,
            alpha=alpha,
            weight_decay=weight_decay,
            momentum=momentum
        )


def get_lr_scheduler(config_dict: typing.Dict, optimizer: nn.Module) -> nn.Module:

    name = config_dict.get("name")

    if name.lower() == 'reducelronplateau':

        factor = config_dict.get("factor")
        reduce_lr = config_dict.get("reduce", "min")
        min_lr = config_dict.get("min_lr")
        patience = config_dict.get("patience")

        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=factor,
            reduce=reduce_lr,
            patience=patience,
            min_lr=min_lr
        )

    if name.lower() == 'steplr':

        step_size = config_dict.get("step_size")
        gamma = config_dict.get("gamma")

        return lr_schedulers.StepLRScheduler(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma
        )
    if name.lower() == "polylr":

        gamma = config_dict.get('gamma')
        max_iters = config_dict.get("max_iters")

        return lr_schedulers.PolyLRScheduler(
            optimizer=optimizer,
            max_iters=max_iters,
            power=gamma
        )
    if name.lower() == "explr":
        gamma = config_dict.get('gamma')
        return lr_schedulers.ExponentialLRScheduler(
            optimizer=optimizer,
            gamma=gamma
        )


def get_efficientnet_network(network_name: str) -> effnet.EfficientNet:
    """
    Loads network, based on the provided name
    Parameters:
    -----------
    lowercased name of the network to load
    Example:
        - "efficientnet-b5"
        - "efficientnet-b7"
    """
    try:
        network: nn.Module = effnet.EfficientNet.from_pretrained(
            network_name.lower())
        final_features = network._fc.in_features
        final_layer = nn.Linear(in_features=final_features, out_features=2)
        network._fc = final_layer
        return network
    except (Exception) as err:
        Logger.error(err)
        raise RuntimeError("Failed to load network")

def get_evaluation_metric_by_name(eval_metric_name: str) -> nn.Module:
    """
    Returns evaluation metric by it's name
    """
    if eval_metric_name.lower() == "f1_score":
        return metrics.F1Score()

    if eval_metric_name.lower() == "precision":
        return metrics.Precision()

    if eval_metric_name.lower() == "recall":
        return metrics.Recall()
    else:
        raise SystemExit("""RuntimeError:
        invalid evaluation metric name provided. 
        Options: 'f1_score', 'recall', 'precision'. But got '%s'""" % eval_metric_name)

def load_image_paths(source_path: str):
    """
    Returns list of image paths, presented
    inside the 'source_path' directory.
    Supported image extensions: "jpeg", "png", "jpg", "avif"
    """
    return [
        os.path.join(source_path, file_path)
        for file_path in os.listdir(source_path)
        if file_path.endswith("jpeg") or file_path.endswith("png")
        or file_path.endswith("jpg") or file_path.endswith("avif")
    ]

def load_deepfake_dataset(
    image_paths: typing.Union[typing.List, numpy.ndarray, torch.Tensor], 
    labels: typing.Union[typing.List, numpy.ndarray, torch.Tensor], 
    **kwargs
):
    """
    Returns DeepfakeDataset abstraction class
    Parameters:
    -----------
    train_image_paths: (list) - list of image path urls
    train_labels: (list) - list of images labels
    augmentations: (albumentations.Compose) - data image augmentations

    Returns:
        - DeepfakeDataset abstraction class
    """

    return datasets.DeepfakeDataset(
        image_paths=image_paths,
        image_labels=labels,
        **kwargs
    )

def load_labels_to_csv(source_path: str):
    """
    Loads labels for the dataset
    from specified source path arg.
    """
    return pandas.read_csv(source_path)


def convert_to_jpeg(input_img: numpy.ndarray):
    success, enc_data = cv2.imencode('jpeg', input_img)
    if not success:
        raise ValueError("failed to convert image to jpeg format")
    return cv2.imdecode(enc_data, cv2.IMREAD_UNCHANGED)

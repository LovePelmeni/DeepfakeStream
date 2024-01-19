import json 
import typing
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.adamw import AdamW
from torch.optim.adamax import Adamax
from torch.optim.rmsprop import RMSprop 
import pandas
import os


def load_config(config_path: str) -> typing.Dict:
    with open(config_path, mode='r') as json_config:
        json_file = json.load(json_config)
    json_config.close()
    return json_file

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

def get_scheduler(config_dict: typing.Dict, optimizer: nn.Module) -> nn.Module:

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

        return lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma
        )

def load_images(source_path: str):
    """
    Loads local .mp4 video or image
    files, presented in the source path
    """
    return [
        os.path.join(source_path, file_path)
        for file_path in os.listdir(source_path)
    ]

def load_labels_to_csv(source_path: str):
    """
    Loads labels for the dataset
    from specified source path arg.
    """
    return pandas.read_csv(source_path) 

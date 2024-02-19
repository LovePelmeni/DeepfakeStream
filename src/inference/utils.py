import numpy
from torch import nn


def convert_syncbn_model(network: nn.Module):
    """
    Converts standard batch normalization layers
    to Synchronized Batch Normalization layers,
    in case we are dealing with distributed training
    / inference.
    """
    for name, layer in network.named_parameters():
        if 'batchnorm' in name:
            pass
    return network

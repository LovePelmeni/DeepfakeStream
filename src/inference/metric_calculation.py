"""
File supplies methods
for calculating number of FLOPs
and MAC for network
"""
from fvcore.nn import flop_count_table, FlopCountAnalysis
from torch import nn
import torch


def calculate_model_flops(network: nn.Module, input_tensor: torch.Tensor):
    flops = FlopCountAnalysis(model=network, inputs=input_tensor)
    print("total model FLOPS: %s" % flops.total())
    print(flop_count_table(flops=flops))

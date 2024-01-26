from torch.nn import Linear, Conv2d, AdaptiveAvgPool2d, Dropout
from torch import nn
import torch
from encoder_params import ENCODER_PARAMS

def initialize_srm_weights(input_channels: int = 3) -> nn.Module: 
    pass

class DeepfakeClassifierSRM(nn.Module):
    """
    Deepfake Classifier model, based on
    SRM (Style-Based Recalibration Module) concept
    for deep noise analysis. 
    """
    def __init__(self, 
        input_channels: int, 
        encoder: nn.Module, 
        dropout_rate: float = 0.5
    ):
        self.encoder = encoder 
        self.srm_layer = initialize_srm_weights(input_channels=input_channels)
        self.avg_pool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout_layer = Dropout(p=dropout_rate)
        self.fc = Linear(in_features=ENCODER_PARAMS[encoder]['in_features'], out_features=1)

    def forward(self, input_image: torch.Tensor):
        noise_features = self.srm_layer(input_image)
        avg_pooled = self.avg_pool(noise_features).flatten(1)
        features = self.dropout_layer(avg_pooled)
        output = self.fc(features)
        return output




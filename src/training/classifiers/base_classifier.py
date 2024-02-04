from torch import nn
import torch
from abc import ABC

ENCODER_CONFIGURATION = {
    "efficientnet-b1": {
        
    }
}

class BaseClassifier(ABC):
    pass

class DeepfakeClassifier(nn.Module, BaseClassifier):
    """
    Deepfake classifier prototype
    implementation, with help of research paper,
    provided under https://assets.researchsquare.com/files/rs-1844392/v1_covered.pdf?c=1659727359

    Parameters:
    -----------
        input_channels: depth of the input image (either 1 or 3)
        encoder - (nn.Module) EfficientNet-like encoder network
    
    NOTE:
        Encoder network should output nxmx1280 feature vector
        so, you have to adjust internal model parameters, to match this 
        requirement.
    """
    def __init__(self, # 256x256
        input_channels: int, 
        encoder: nn.Module
    ):
        super(DeepfakeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels)
        self.encoder = encoder
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout()
        self.dense1 = nn.Linear(in_features=1280, out_features=128)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features=128, out_features=64)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_map: torch.Tensor):
        output = self.conv1(input_map)
        output = self.encoder(output)
        output = self.avgpool1(output)
        output = self.dropout1(output)
        output = self.dense1(output)
        output = self.relu1(output)
        output = self.dense2(output)
        output = self.relu2(output)
        output = self.dense3(output)
        output_prob = self.sigmoid(output)
        return output_prob
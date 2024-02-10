from torch import nn
import torch
from src.training.classifiers import srm_conv
from timm import models

encoders = {
    "efficientnet-b3": {
        "cls": models.efficientnet.efficientnet_b3,
        "features": 1280
    },
    "efficientnet-b2": {
        "cls": models.efficientnet.efficientnet_b2,
        "features": 1280,
    },
    "efficientnet-b4": {
        "cls": models.efficientnet.efficientnet_b1,
        "features": 1280
    }
}

class DeepfakeClassifier(nn.Module):
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
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=input_channels,
            bias=False
        ) 
        self.encoder = encoder
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout()
        self.dense1 = nn.Linear(
            in_features=encoders[encoder]['out_features'],
            out_features=128,
            bias=True
        )
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(
            in_features=128, 
            out_features=64,
            bias=True
        )
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(
            in_features=64, 
            out_features=1,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_map: torch.Tensor):
        output = self.conv1(input_map)
        output = self.encoder(output)
        output = self.avgpool1(output).flatten(1)
        output = self.dropout1(output)
        output = self.dense1(output)
        output = self.relu1(output)
        output = self.dense2(output)
        output = self.relu2(output)
        output = self.dense3(output)
        output_prob = self.sigmoid(output)
        return output_prob

class DeepfakeClassifierSRM(nn.Module):
    """
    Version of deepfake classifier, based on concept
    of SRM (Spatial Rich Model) Filters.
    """
    def __init__(self, input_channels: int, encoder):
        super(DeepfakeClassifier, self).__init__(encoder)

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=input_channels,
            stride=1,
            bias=False
        )
        self.encoder = encoder
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.srm_conv = srm_conv.SRMConv(in_channels=input_channels)
        self.dropout1 = nn.Dropout()
        self.dense1 = nn.Linear(
            in_features=1280, 
            out_features=128, 
            bias=True
        )
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(
            in_features=128,
            out_features=64, 
            bias=True
        )
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(
            in_features=64, 
            out_features=1, 
            bias=True
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_map: torch.Tensor):
        noise = self.srm_conv(input_map)
        output = self.conv1(noise)
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
class GlobalWeightedAveragePooling(nn.Module):
    """
    Global weighted average pooling, which examinates
    better pixel-wise localization. 
    Paper: "Global Weighted Average Pooling Bridges
    Pixel-level Localization and Image-level
    Classification by Suo Qiu"
    """
    def __init__(self, input_size: int, flatten: bool = False):
        super(GlobalWeightedAveragePooling, self).__init__()
        self.conv = nn.Conv2d(
            input_size, 
            out_channels=1, 
            kernel_size=1, 
            bias=False
        )
        self.flatten = flatten

    def fscore(self, x):
            m = self.conv(x)
            m = m.sigmoid().exp()
            return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x
class DeepfakeClassifierGWAP(nn.Module):
    """
    Implementation of the Deepfake classifier
    with usage of Global Weighted Average Pooling layer,
    instead of standard avg pool.
    """
    def __init__(self, encoder):
        super(DeepfakeClassifierGWAP, self).__init__()

        self.encoder = encoders[encoder]['cls']()

        self.globalpool1 = GlobalWeightedAveragePooling(
            input_size=encoders[encoder]['features'],
            flatten=True
        )

        self.batchnorm1 = nn.BatchNorm1d(
            num_features=encoders[encoder]['features']
        )

        self.dropout1 = nn.Dropout(p=0.5)

        self.dense1 = nn.Linear(
            in_features=encoders[encoder]['features'], 
            out_features=64,
            bias=True
        )

        self.relu1 = nn.ReLU()

        self.dense2 = nn.Linear(
            in_features=64,
            out_features=1,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image: torch.Tensor):

        if not len(image.shape) == 4:
            raise ValueError("""input should have 4 dimensions: 
            (batch_size, channels, height, width), however, 
            you passed data with %s dimensions.""" % len(image.shape))

        output = self.encoder(image)
        output = self.globalpool1(output)
        output = self.batchnorm1(output)
        output = self.dropout1(output)
        output = self.dense1(output)
        output = self.relu1(output)
        output = self.dense2(output)
        probs = self.sigmoid(output)
        return probs

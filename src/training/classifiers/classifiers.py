from torch import nn
import torch
from src.training.classifiers import srm_conv
from timm.models.efficientnet import (
    tf_efficientnetv2_b3,
    tf_efficientnetv2_b2,
)
from functools import partial 

encoder_params = {
    "tf_efficientnet_b3": {
        "features": 1536,
        "encoder": partial(tf_efficientnetv2_b3, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b2": {
        "features": 1408,
        "encoder": partial(tf_efficientnetv2_b2, pretrained=False, drop_path_rate=0.2)
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
        encoder_name: str
    ):
        super(DeepfakeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=input_channels,
            bias=False
        ) 
        self.encoder  = encoder_params[encoder_name]['encoder']()
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout()
        self.dense1 = nn.Linear(
            in_features=encoder_params[encoder_name]['features'],
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
        output = self.encoder.forward_features(output)
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
    def __init__(self, 
        input_channels: int, 
        encoder_name: str, 
        num_classes: int,
        dropout_rate: float = 0.5
    ):
        super(DeepfakeClassifierSRM, self).__init__()
        self.encoder_name = encoder_name
        self.srm_conv = srm_conv.SRMConv(in_channels=input_channels)
        self.encoder = encoder_params[encoder_name]['encoder']()
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1)) # x x 1 x 1
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dense1 = nn.Linear(
            in_features=encoder_params[encoder_name]['features'], 
            out_features=64, 
            bias=True,
        )
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(
            in_features=64,
            out_features=32, 
            bias=True
        )
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(
            in_features=32,
            out_features=num_classes,
            bias=True
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_map: torch.Tensor):
        noise = self.srm_conv(input_map)
        output = self.encoder.forward_features(noise)
        output = self.avgpool1(output).view(-1, encoder_params[self.encoder_name]['features'])
        output = self.dropout1(output)
        output = self.dense1(output)
        output = self.relu1(output)
        output = self.dense2(output)
        output = self.relu2(output)
        output = self.dense3(output)
        output_probs = self.softmax(output)
        return output_probs
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
    def __init__(self, encoder_name: str):
        super(DeepfakeClassifierGWAP, self).__init__()

        self.encoder = encoder_params[encoder_name]['encoder']()

        self.globalpool1 = GlobalWeightedAveragePooling(
            input_size=encoder_params[encoder_name]['features'],
            flatten=True
        )

        self.batchnorm1 = nn.BatchNorm1d(
            num_features=encoder_params[encoder_name]['features']
        )

        self.dropout1 = nn.Dropout(p=0.5)

        self.dense1 = nn.Linear(
            in_features=encoder_params[encoder_name]['features'], 
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

        output = self.encoder.forward_features(image)
        output = self.globalpool1(output)
        output = self.batchnorm1(output)
        output = self.dropout1(output)
        output = self.dense1(output)
        output = self.relu1(output)
        output = self.dense2(output)
        probs = self.sigmoid(output)
        return probs
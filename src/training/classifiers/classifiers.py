from torch import nn
import torch
from src.training.classifiers import srm_conv
from timm.models.efficientnet import (
    tf_efficientnetv2_b3,
    tf_efficientnetv2_b2,
    _cfg
)
from torchvision.models import (
    resnet50,
    resnet101,
    ResNet50_Weights,
    ResNet101_Weights
)

from functools import partial
import typing


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

resnet_encoder_params = {
    "resnet_101": {
        "features": 2048,
        "encoder": partial(resnet101, weights=ResNet101_Weights),
    },
    "resnet_50": {
        "features": 2048,
        "encoder": partial(resnet50, weights=ResNet50_Weights)
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
        encoder_name - (nn.Module) EfficientNet-like encoder network

    NOTE:
        Encoder network should output nxmx1280 feature vector
        so, you have to adjust internal model parameters, to match this 
        requirement.
    """

    def __init__(self,  # 256x256
                 input_channels: int,
                 encoder_name: str,
                 device: str = 'cpu'
                 ):
        super(DeepfakeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            bias=False
        ).to(device)
        self.encoder = encoder_params[encoder_name]['encoder']()
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.dropout1 = nn.Dropout().to(device)
        self.dense1 = nn.Linear(
            in_features=encoder_params[encoder_name]['features'],
            out_features=128,
            bias=True,
            device=device
        )
        self.relu1 = nn.ReLU().to(device)
        self.dense2 = nn.Linear(
            in_features=128,
            out_features=64,
            bias=True,
            device=device
        )
        self.relu2 = nn.ReLU().to(device)
        self.dense3 = nn.Linear(
            in_features=64,
            out_features=1,
            bias=True,
            device=device
        )
        self.sigmoid = nn.Sigmoid().to(device)

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
                 input_channels: int = 3,
                 encoder_name: str = list(encoder_params.keys())[-1],
                 num_classes: int = 2,
                 encoder_pretrained_config: typing.Dict = None,
                 dropout_rate: float = 0.5,
                 device: str = 'cpu'
                 ):
        super(DeepfakeClassifierSRM, self).__init__()

        # preping custom configuration for the EfficientNet encoder, in case presented
        if encoder_pretrained_config is not None:
            pretrained_cfg = _cfg(
                url=encoder_pretrained_config.get("url", ''),
                input_size=encoder_pretrained_config.get(
                    "encoder_input_image_size"),
                file=encoder_pretrained_config.get(
                    "encoder_weights_path", None)
            )
        else:
            pretrained_cfg = None

        self.encoder_name = encoder_name
        self.srm_conv = srm_conv.SRMConv(in_channels=input_channels).to(device)
        self.encoder = encoder_params[encoder_name]['encoder'](
            pretrained_cfg=_cfg(pretrained_cfg))
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1)).to(device)  # x x 1 x 1
        self.dropout1 = nn.Dropout(p=dropout_rate).to(device)
        self.dense1 = nn.Linear(
            in_features=encoder_params[encoder_name]['features'],
            out_features=64,
            bias=True,
            device=device
        )
        self.relu1 = nn.ReLU().to(device)
        self.dense2 = nn.Linear(
            in_features=64,
            out_features=32,
            bias=True,
            device=device
        )
        self.relu2 = nn.ReLU().to(device)
        self.dense3 = nn.Linear(
            in_features=32,
            out_features=num_classes,
            bias=True,
            device=device
        )
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, input_map: torch.Tensor):
        noise = self.srm_conv(input_map)
        output = self.encoder.forward_features(noise)
        output = self.avgpool1(
            output).view(-1, encoder_params[self.encoder_name]['features'])
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

    Parameters:
    -----------
        encoder_name - name of the encoder CNN-based 
        network from 'encoder_params'
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

class DistilDeepfakeClassifierSRM(nn.Module):
    """
    Implementation of the light weight
    encoder network with reduced number of parameters.
    This architecture is designed to serve as a distilled
    version of the original DeepfakeClassifierSRM.
    
    Parameters:
    -----------
        encoder_name - name of the resnet encoder.
        dropout_prob - dropout probability.
        num_classes - number of output classes.
        input_channels - number of channels in the input images. 
        either 1 (for grayscale or binary images) or 3 (for RGB images).

    NOTE:
        the output of this model contains raw logits.
        If you want to convert them to probability distribution,
        make sure to apply nn.Softmax(dim=-1).
    """
    def __init__(self, 
        input_channels: int, 
        encoder_name: str, 
        num_classes: int,
        dropout_prob: float = 0.5
    ):
        super(DistilDeepfakeClassifierSRM, self).__init__()
        self.encoder_name = encoder_name
        self.encoder_out_features = resnet_encoder_params[encoder_name]['features']

        self.srm_conv = srm_conv.SRMConv(in_channels=input_channels)
        self.encoder = resnet_encoder_params[encoder_name]['encoder']()

        if not hasattr(self.encoder, 'fc'):
            raise ValueError(
                """invalid feature extraction model.
                 Should be torchvision.models
                """
            )
        else:
            self.encoder.fc = nn.Identity()

        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_block1 = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(
                in_features=self.encoder_out_features,
                out_features=self.encoder_out_features//2,
                bias=True
            ),
            nn.BatchNorm1d(num_features=self.encoder_out_features//2),
            nn.ReLU()
        )
        self.dense_block2 = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(
                in_features=self.encoder_out_features//2,
                out_features=self.encoder_out_features//4,
                bias=True
            ),
            nn.BatchNorm1d(
                num_features=self.encoder_out_features//4, 
                track_running_stats=True
            ),
            nn.ReLU()
        )
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.proj_head = nn.Linear(
            in_features=self.encoder_out_features//4,
            out_features=num_classes,
            bias=False
        )

    def forward(self, input_images: torch.Tensor):
        out_srm_features = self.srm_conv(input_images)
        encoder_features = self.encoder(out_srm_features)
        pooled_features = self.avg_pool1(encoder_features)
        pooled_features = self.dense_block1(pooled_features)
        pooled_features = self.dense_block2(pooled_features)
        pooled_features = self.dropout3(pooled_features)
        proj_features = self.proj_head(pooled_features)
        return proj_features

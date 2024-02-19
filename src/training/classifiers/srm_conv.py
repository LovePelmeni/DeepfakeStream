from torch import nn
import torch
import numpy


class SRMConv(nn.Module):
    """
    5x5 3-channel SRM Filter for Noise Analysis
    Paper for reference: https://www.cs.columbia.edu/~jrk/NSFgrants/videoaffinity/Interim/22x_Rohit.pdf

    NOTE:
        SRM Convolution has only
        two implementations: 3x3 and 5x5
        it is not recommended to change the size
        of the layer, as it may lead to unexpected results
    """

    def __init__(self, in_channels: int):
        super(SRMConv, self).__init__()

        with torch.no_grad():

            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=5,
                stride=(1, 1),
                padding=2,
                bias=False
            )

            self.conv.weight = nn.Parameter(
                data=torch.from_numpy(numpy.array(
                        [
                            [  # srm 1/2 horiz
                                [0., 0., 0., 0., 0.],  # noqa: E241,E201
                                [0., 0., 0., 0., 0.],  # noqa: E241,E201
                                [0., 1., -2., 1., 0.],  # noqa: E241,E201
                                [0., 0., 0., 0., 0.],  # noqa: E241,E201
                                [0., 0., 0., 0., 0.],  # noqa: E241,E201
                            ], [  # srm 1/4
                                [0., 0., 0., 0., 0.],  # noqa: E241,E201
                                [0., -1., 2., -1., 0.],  # noqa: E241,E201
                                [0., 2., -4., 2., 0.],  # noqa: E241,E201
                                [0., -1., 2., -1., 0.],  # noqa: E241,E201
                                [0., 0., 0., 0., 0.],  # noqa: E241,E201
                            ], [  # srm 1/12
                                [-1., 2., -2., 2., -1.],  # noqa: E241,E201
                                [2., -6., 8., -6., 2.],  # noqa: E241,E201
                                [-2., 8., -12., 8., -2.],  # noqa: E241,E201
                                [2., -6., 8., -6., 2.],  # noqa: E241,E201
                                [-1., 2., -2., 2., -1.],  # noqa: E241,E201
                            ]
                        ]
                    )
                ),
                requires_grad=False
            )

    def forward(self, input_map: torch.Tensor):
        return self.conv(input_map.squeeze(0))

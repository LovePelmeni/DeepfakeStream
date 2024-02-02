import numpy
from torch import nn
import torch
import typing 

"""
File, containing basic evaluation metrics 
for testing DeepFake detection algorithm during validation phase
"""
class F1Score(nn.Module):

    def forward(self,
                predicted_values: numpy.ndarray,
                actual_labels: numpy.ndarray,
                smooth: float = 0.00001
                ):
        tp = len(predicted_values == actual_labels)
        fp = len(predicted_values == 1 != actual_labels)
        fn = len(predicted_values == 0 != actual_labels)
        precision = tp / (tp + fn)
        recall = tp / (tp + fp)
        return 2 * (precision * recall) / (precision + recall) + smooth

class IOUScore(nn.Module):
    """
    Implementation of the intersection
    over union metric

    Parameters:
    -----------
        eps - smoothing factor
    """
    def __init__(self, eps: float = 1e-7):
        super(IOUScore, self).__init__()
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, actual_boxes: torch.Tensor):

        pred_boxes = pred_boxes.float()
        actual_boxes = actual_boxes.float()

        ymin = torch.max(pred_boxes[:, 1], actual_boxes[:, 1])
        xmin = torch.max(pred_boxes[:, 0], actual_boxes[:, 0])
        xmax = torch.min(pred_boxes[:, 2], actual_boxes[:, 2])
        ymax = torch.min(pred_boxes[:, 3], actual_boxes[:, 3])
        
        pred_area = (pred_boxes[:, 3] - pred_boxes[:, 0]) * (pred_boxes[:, 2] - pred_boxes[:, 1])
        act_area = (actual_boxes[:, 3] - actual_boxes[:, 0]) * (actual_boxes[:, 2] - actual_boxes[:, 1])
        union_area = pred_area + act_area
        intersection_area = (xmax - xmin) * (ymax - ymin)
        return intersection_area / union_area + self.eps

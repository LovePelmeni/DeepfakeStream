from torch import nn
import torch

class EarlyStopping(object):
    """
    Class for regularizing 
    network training process
    """

    def __init__(self, patience: int, min_diff: float):
        self.patience = patience
        self.min_diff = min_diff
        self.base_metric = None
        self.current_patience = patience

    def step(self, metric: float):

        if self.base_metric is None:
            self.base_metric = metric

        elif self.base_metric >= metric:
            self.current_patience -= 1
        else:
            self.base_metric = metric
            self.current_patience = self.patience
        return (self.current_patience == 0)

class LabelSmoothing(nn.Module):

    def __init__(self, 
        total_classes: int, 
        eps: float = 0, 
        min_confidence: float = 0
    ):
        self.eps = eps 
        self.min_confidence: float = min_confidence
        self.total_classes: int = total_classes

    def forward(self, 
        input_labels: torch.Tensor, 
        pred_probs: torch.Tensor
    ):
        one_hot_labels = torch.stack([
            torch.full(size=self.total_classes, 
            fill_value=self.eps / (self.total_classes - 1)) 
            for _ in range(input_labels.shape[0])
        ])
        for pred in range(len(pred_probs)):
            max_index = torch.argmax(pred_probs[pred], axis=0)
            one_hot_labels[pred][max_index] = (pred_probs[pred][max_index] - self.eps)
        return -torch.sum(one_hot_labels * torch.log2() + (1 - one_hot_labels) * torch.log2(1 - pred_probs))
        



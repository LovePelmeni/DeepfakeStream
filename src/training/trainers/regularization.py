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

    def __init__(self, etta: float):
        super(LabelSmoothing, self).__init__(etta)

        self.etta = etta
        self.smoother = lambda logits: (
            1 - etta) * logits + etta * (1 / len(logits) + 1)

    def forward(self, dist_logits: torch.Tensor):
        smoothed_logits = self.smoother(dist_logits)
        return smoothed_logits




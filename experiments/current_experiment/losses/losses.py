import numpy
from torch import nn

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss
    function, which is highly beneficial
    to use in imbalanced classification systems

    """
    def __init__(self, gamma: float):
        self.gamma = gamma 
    
    def forward(self, predicted_prob, actual_label):
        bce_loss = (1 - actual_label) * numpy.log2(predicted_prob)
        return -((1 - predicted_prob) ** self.gamma) * bce_loss

import numpy
import torch
from torch import nn
from torch.nn import functional

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss
    function, which is highly beneficial
    to use in imbalanced classification systems

    """
    def __init__(self, gamma: float):
        self.gamma = gamma 
    
    def forward(self, predicted_prob: torch.Tensor, actual_label: torch.Tensor):
        pred_probs = torch.argmax(predicted_prob, axis=1).flatten()
        actual_labels = torch.argmax(actual_label, axis=1).flatten()
        bce_loss = functional.binary_cross_entropy_with_logits(pred_probs, actual_labels)
        return -((1 - predicted_prob) ** self.gamma) * bce_loss





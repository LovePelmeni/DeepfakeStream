import torch
from torch import nn
from torch.nn import functional

class WeigthedLoss(nn.Module):
    """
    Implementation of weighted
    loss function, which considers 
    individual weights for different classes
    """
    def __init__(self, 
        loss_function: nn.Module,
        weights: torch.Tensor, 
        weight_type: str
    ):
        self.weights = weights.to(weight_type)
        self.loss_function = loss_function

    def forward(self, pred_labels: torch.Tensor, true_labels: torch.Tensor):
        total_loss = 0
        for idx, pred_label in enumerate(pred_labels):
            weight = self.weights[pred_label]
            loss = self.loss_function(pred_label, true_labels[idx])
            total_loss += (loss * weight)
        return total_loss
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



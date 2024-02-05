import torch
from torch import nn
from torch.nn import functional
import typing

class WeightedLoss(nn.Module):
    """
    Implementation of weighted
    loss function, which considers 
    individual weights for different classes
    """
    def __init__(self, 
        loss_function: nn.Module,
        weights: list, 
        weight_type: torch.dtype
    ):
        super(WeightedLoss, self).__init__()
        self.weights = torch.as_tensor(weights).to(weight_type)
        self.loss_function = loss_function

    def forward(self, pred_probs: torch.Tensor, true_labels: torch.Tensor):
        """
        Performs forward pass of the weighted loss function.

        NOTE:
            - pred_probs expected to be a torch.Tensor object 
            of predicted class probabilities (0, 1, background)
            Example:
                - [[0.4, 0.5, 0.7], [0.5, 0.25, 0.25]]
        """
        total_loss = 0

        output_labels = torch.argmax(pred_probs, dim=1, keepdim=False)
        output_probs = torch.max(pred_probs, dim=1, keepdim=False)
        
        for idx, pred_label in enumerate(output_labels):

            weight = self.weights[pred_label]

            loss = self.loss_function(
                torch.as_tensor([output_probs[idx]]), 
                torch.as_tensor([true_labels[idx].float()]),
            )
            
            total_loss += (loss * weight)
        return total_loss

class FocalLoss(nn.Module):

    def __init__(self, weights: list, weight_type: torch.dtype, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        weight_values = torch.as_tensor(weights).to(weight_type)
        weight_classes = range(len(weights))
        self.weights = {c:w for c, w in zip(weight_classes, weight_values)}

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Flatten the predictions and ground truth masks
        
        pred_labels = torch.argmax(y_pred.float(), dim=1, keepdim=False).to(torch.float32)
        label_weights = torch.as_tensor([self.weights[label.item()] for label in pred_labels])
        pred_probs = torch.amax(y_pred.float(), 1, False).to(torch.float32)

        # Calculate binary cross-entropy loss
        bce_loss = functional.binary_cross_entropy_with_logits(
            input=pred_probs, target=y_true.float(), 
            pos_weight=label_weights,
            reduction='none'
        )

        # Calculate focal loss
        focal_loss = (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss

        # Calculate mean over all elements
        mean_loss = torch.mean(focal_loss)

        return mean_loss

class CIOULoss(nn.Module):
    """
    Complete Intersection-Over-Union
    implementation, according to the paper
    "Faster and Better Learning for Bounding Box Regression by Z Zheng"
    provides better generalization, avoids vanishing gradient problem,

    considers important geometrical factors: 
        aspect ratio, 
        normalized distance between box centers,
        overlap area
    """
    def __init__(self, 
        reduction: typing.Literal['sum', 'mean' 'none'] = 'none', 
        eps: float = 1e-7
    ):
        super(CIOULoss, self).__init__()
        self.eps = eps 
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, actual_boxes: torch.Tensor):

        pred_float = pred_boxes.float()
        actual_float = actual_boxes.float()
        
        inter_xmin = torch.max(pred_float[:, 0], actual_float[:, 0])
        inter_ymin = torch.max(pred_float[:, 1], actual_float[:, 1])
        inter_xmax = torch.max(pred_float[:, 2], actual_float[:, 2])
        inter_ymax = torch.max(pred_float[:, 3], actual_float[:, 3])
        
        inter_area = (
            torch.clamp(inter_xmax - inter_xmin, min=0) * 
            torch.clamp(inter_ymax - inter_ymin, min=0)
        )

        pred_area = (
            (pred_boxes[:, 2] - pred_boxes[:, 0]) * 
            (pred_boxes[:, 3] - pred_boxes[:, 1])
        )
        
        actual_area = (
            (actual_boxes[:, 2] - actual_boxes[:, 0]) * 
            (actual_boxes[:, 3] - actual_boxes[:, 1]) 
        )

        union_area = pred_area + actual_area
        iou = inter_area / (pred_area + actual_area + self.eps)

        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) // 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) // 2
        true_center_x = (actual_boxes[:, 0] + actual_boxes[:, 2]) // 2
        true_center_y = (actual_boxes[:, 1] + actual_boxes[:, 3]) // 2
        
        center_distance = ((pred_center_x - true_center_x) ** 2 + (pred_center_y - true_center_y) ** 2) ** 0.5 
        
        ar_gt = (actual_float[:, 2] - actual_float[:, 0]) / (actual_float[:, 3] - actual_float[:, 1])
        ar_pr = (pred_float[:, 2] - pred_float[:, 0]) / (pred_float[:, 3] - pred_float[:, 1])

        v = (4 / torch.pi ** 2) * torch.pow((torch.atan(ar_gt) - torch.atan(ar_pr)), 2)
        alpha = v / (1 - iou + v)

        ciou_loss = iou - (center_distance ** 2) / (2 * union_area) - alpha * v
        return torch.mean(ciou_loss)



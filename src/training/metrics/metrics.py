import numpy
from torch import nn
import typing 
import torch

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


class AveragePrecision(nn.Module):
    """
    Implementation of AP (Average Precision)
    Evaluation Metric, commonly used for evaluating
    video object detection systems
    """
    def __init__(self, 
        total_classes: int,
        conf_threshold: float = 0.5, 
        iou_threshold: float = 0.5,
    ):
        super(AveragePrecision, self).__init__()

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.total_classes = total_classes
        self.iou = IOUScore()

    def forward(self, 
        pred_boxes: typing.Dict[str, typing.Dict], 
        actual_boxes: typing.Dict[str, typing.Dict]
     ): 
        """
        Parameters:
        -----------
        pred_boxes - dictionary of following format: [{'img1': {'boxes': [], 'conf_scores': []}}, ...]
        actual_boxes - list of following objects: [{'img1': {'boxes': []}}, {'img2': {'boxes': []}}
        """
        
        # matching similar boxes, based on IOU score

        total_boxes = torch.sum([len(actual_boxes[sample]['boxes']) for sample in actual_boxes])
        iou_scores = torch.zeros(total_boxes)

        for img_id in pred_boxes:

            pred_img_boxes = pred_boxes[img_id]
            actual_img_boxes = actual_boxes[img_id]

            for box1 in range(len(pred_img_boxes)):
                iou_scores = [
                    self.iou(pred_img_boxes[box1], actual_img_boxes[actual_box])
                    for actual_box in actual_img_boxes
                ]
                iou_scores[box1] = torch.argmax(iou_scores, dim=0)
                
        # calculating true positives and false positives
        conf_scores = torch.where(conf_scores >= self.conf_threshold, 1, 0)
        iou_scores = torch.where(iou_scores >= self.iou_threshold, 1, 0)
        overall_box_status = numpy.multiply(conf_scores, iou_scores)
        
        tp = numpy.where(overall_box_status != 0, 1, 0)

        recalls = torch.cumsum(tp / torch.sum(tp))
        precisions = torch.cumsum(tp / len(tp))

        pr_area = torch.trapz(y=precisions, x=recalls)
        return pr_area
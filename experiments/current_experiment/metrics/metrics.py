import numpy 
from torch import nn

"""
File, containing basic evaluation metrics 
for testing DeepFake detection algorithm during validation phase
"""

class Precision(nn.Module):
    """
    Basic implementation of the precision
    evaluation metric
    """
    def forward(self, 
        pred_labels: numpy.ndarray, 
        actual_labels: numpy.ndarray
    ):
        pass

class Recall(nn.Module):
    """
    Basic implementation of the Recall
    evaluation metric
    """
    def forward(self, 
        pred_labels: numpy.ndarray, 
        actual_labels: numpy.ndarray
    ):
        pass

class F1Score(nn.Module):

    def forward(self, 
        predicted_values: numpy.ndarray, 
        actual_labels: numpy.ndarray,
        smooth: float = 0.00001
    ):
        tp = len(predicted_values == actual_labels)
        fp = len(predicted_values == 1 != actual_labels)
        fn = len(predicted_values == 0 != actual_labels)
        precision =  tp / (tp + fn)
        recall = tp / (tp + fp)
        return 2 * (precision * recall) / (precision + recall) + smooth

class RocAuc(nn.Module):

    def forward(self, 
        predicted_labels: numpy.ndarray, 
        true_labels: numpy.ndarray
    ):
        areas = []
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        area = numpy.trapz()
        areas.append(area)
        return numpy.mean(areas)

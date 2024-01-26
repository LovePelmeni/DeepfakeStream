import torch
from torch import nn
import matplotlib.pyplot as plt
import typing
import seaborn as sns
from torch.utils import data
import logging

Logger = logging.getLogger("sliced_eval_logger")


class SlicedEvaluation(object):

    """
    Class for evaluating model on different 
    sliced, splitted based on the certain criteria (class label or other fields)
    This ensures, that network does not discriminate
    objects, based on their properties and guarantees it's robustness.
    """

    visual_palette: typing.ClassVar = ["green", "black"]

    def __init__(self,
                 network: nn.Module,
                 inf_device: typing.Literal['cuda', 'mps', 'cpu'],
                 eval_metric: nn.Module
                 ):
        self.network = network.to(inf_device)
        self.eval_metric = eval_metric
        self.inf_device = inf_device

    @classmethod
    def visualize_sliced_evals(cls, input_metrics: typing.Dict):

        plt.figure(figsize=(20, 5))
        sns.barplot(
            y=list(input_metrics.values()),
            x=[0, 1],
            hue=list(input_metrics.values()),
            palette=sns.color_palette(cls.visual_palette)
        )
        plt.legend(loc='upper right')
        plt.ylim(0, 1)
        plt.xlim(0, len(input_metrics))
        plt.tick_params(axis='x', rotation=30)

    def _evaluate_batch(self,
                        images: torch.Tensor,
                        true_labels: torch.Tensor
                        ):
        predictions = self.network.forward(images.clone().detach()).cpu()

        pred_labels = torch.as_tensor([
            torch.argmax(predictions[idx], axis=0)
            for idx in range(len(predictions))
        ])
        binary_labels = (pred_labels == true_labels).to(torch.uint8)
        return self.eval_metric(torch.ones_like(true_labels), binary_labels)

    def evaluate(self, dataset: data.Dataset):

        try:
            output_metrics: typing.Dict[int, float] = {}
            input_images, input_labels = zip(*dataset)

            input_images = torch.stack(input_images)
            input_labels = torch.as_tensor(input_labels)
            unique_classes = torch.unique(input_labels)

        except (Exception) as err:
            Logger.error(err)

        try:

            for class_ in unique_classes:

                indices = torch.where(input_labels == class_)[0].tolist()

                # loading images and label data
                cls_images = input_images[indices].to(self.inf_device)
                cls_labels = input_labels[indices]

                output_metric = self._evaluate_batch(
                    images=cls_images,
                    true_labels=cls_labels
                )
                output_metrics[class_.item()] = round(output_metric, 3)

        except (Exception) as err:
            Logger.error(err)

        return output_metrics

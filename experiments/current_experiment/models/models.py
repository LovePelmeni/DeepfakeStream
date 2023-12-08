import torch
import torchsummary
from torch import nn
from torch import optim
import logging
import torchsummary
import typing
from torch.optim import lr_scheduler
from torchvision import models
from datasets import datasets
from torch.utils import data
import numpy
from regularization import EarlyStopping


logger = logging.getLogger("model_inference")
handler = logging.FileHandler("inference.log")
logger.addHandler(handler)

class DeepfakeClassifier(nn.Module):
    """
    Classification model for recognizing between
    deepfakes of human face and real world human faces
    """

    def __init__(self,
                 loss_function: nn.Module,
                 eval_metric: nn.Module,
                 learning_rate: int,
                 max_epochs: int,
                 checkpoint_per_epoch: int,
                 early_start: int,
                 early_dataset: datasets.DeepFakeClassificationDataset,
                 min_metric_diff: int,
                 early_patience: int,
                 training_device: typing.Literal['cpu', 'cuda', 'mps'] = 'cpu',
                 weight_decay: float = 1,
                 
                 ):
        self.training_device = torch.device(training_device)

        self.early_stopper = EarlyStopping(
            patience=early_patience, 
            min_diff=min_metric_diff
        )

        self.early_start = early_start
        self.early_dataset = early_dataset

        self.network = models.resnet101(
            weights=models.ResNet101_Weights).to(self.training_device)

        self.optimizer = optim.Adam(
            lr=learning_rate,
            params=self.network.parameters(),
            weight_decay=weight_decay
        )

        self.lr_scheduler = lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=10,
        )
        
        self.loss_function = loss_function 
        self.eval_metric = eval_metric

        self.checkpoint_per_epoch = checkpoint_per_epoch
        self.max_epochs = max_epochs

    def save_checkpoint(self, epoch: int, loss: float):
        try:
            torch.save(
                obj={
                    'epoch': epoch,
                    'loss': loss,
                    'optimizer_state': self.optimizer.state_dict().__dict__,
                    'network_state': self.network.state_dict().__dict__,
                }
            )
        except (FileNotFoundError, Exception) as err:
            logger.warning("failed to save checkpoint for the model")
            logger.warning(err)

    def train(self, images: datasets.DeepFakeClassificationDataset):

        if not numpy.all(a=[img.shape[0] == self.input_shape for img in images.images]):
            raise ValueError('All images should match specified input shape')

        loader = data.DataLoader(
            images,
            batch_size=self.batch_size
        )

        best_class_loss = 1 
        losses = []

        for epoch in range(self.max_epochs):
            epoch_losses = []

            for labels, images in loader:

                predictions = self.network.forward(
                    images.to(self.training_device)).cpu()

                loss = self.loss_function(predictions, labels)

                loss.backward()
                epoch_losses.append(loss.item())
                losses.append(numpy.mean(epoch_losses))

                self.optimizer.step()
        
            self.lr_scheduler.step(epoch=epoch)
            best_class_loss = min(best_class_loss, loss.item())

            if epoch % self.checkpoint_per_epoch == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    loss=numpy.mean(epoch_losses)
                )
            if (epoch + 1) >= self.early_start:
                metric = self.evaluate(dataset=self.early_dataset)
                stop_status = self.early_stopper.step(metric=metric)
                if stop_status: break

        return best_class_loss, losses

    def evaluate(self, dataset: datasets.DeepFakeClassificationDataset):

        if not numpy.all(
            a=[img.shape[0] == self.input_shape 
            for img in images.images
            ]
        ):
            raise ValueError(
            'All images should match specified input shape'
            )

        loader = data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size
        )

        with torch.no_grad():
            metrics = []
            for labels, images in loader:
                predictions = self.network.forward(
                    images.to(self.training_device)).cpu()

                metric = self.eval_metric(predictions, labels)
                metrics.append(metric.item())
            return numpy.mean(metrics)

    def save_model(self, test_input, model_path):
        torch.onnx.export(model=self.network, args=test_input, f=model_path)

    def predict(self, images: datasets.DeepFakeClassificationDataset):
        predicted_classes = []
        for img in images:
            predicted_class = numpy.argmax(
                self.network.forward(img)
            )
            predicted_classes.append(predicted_class)
        return predicted_classes

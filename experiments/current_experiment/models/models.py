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


logger = logging.getLogger("model_inference")
handler = logging.FileHandler("inference.log")
logger.addHandler(handler)


def pick_best_batch_size(model: nn.Module, input_size: tuple, device):
    """
    Function picks best estimated
    batch size for model inference,
    with consideration of available computational resources

    Args:
        - model (nn.Module) - model for estimating 
        - input_size (int) - size of the input image for the model.
          Example: (512, 512, 3) 512x512 x (3 channels)
        - device - device for training (either 'cpu', 'gpu' or 'mps')
    """
    summary = torchsummary.summary(
        model,
        input_size=input_size,
        device=device
    ).__dict__
    optimal_batch_size = (summary['available_gpu_bytes'] -
                          model['parameters']) / (summary['for_back_ward_size'])
    return 2 ** torch.ceil(
        torch.log2(optimal_batch_size)
    )


class DeepfakeClassifier(nn.Module):
    """
    Classification model for recognizing between
    deepfakes of human face and real world human faces
    """

    def __init__(self,
                 learning_rate: int,
                 max_epochs: int,
                 input_shape: tuple,
                 checkpoint_per_epoch: int,
                 training_device: typing.Literal['cpu', 'cuda', 'mps'] = 'cpu',
                 weight_decay: float = 1,
                 ):
        self.training_device = torch.device(training_device)
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

        self.input_shape = input_shape
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

        optimal_batch_size = pick_best_batch_size(
            model=self.network,
            input_size=self.input_shape,
        )
        loader = data.DataLoader(
            images,
            batch_size=optimal_batch_size
        )
        losses = []
        for epoch in range(self.max_epochs):
            epoch_losses = []
            for labels, images in loader:
                predictions = self.network.forward(
                    images.to(self.training_device)).cpu()
                loss = self.loss_function(predictions, labels)

                loss.backward()
                epoch_losses.append(loss.item())
                self.optimizer.step()
                self.lr_scheduler.step(epoch=epoch)

            if epoch % self.checkpoint_per_epoch == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    loss=numpy.mean(epoch_losses)
                )
        return numpy.mean(losses)

    def evaluate(self, images: datasets.DeepFakeClassificationDataset):

        if not numpy.all(a=[img.shape[0] == self.input_shape for img in images.images]):
            raise ValueError('All images should match specified input shape')

        optimal_batch_size = pick_best_batch_size(
            model=self.network,
            input_size=self.input_shape
        )

        loader = data.DataLoader(
            images,
            batch_size=optimal_batch_size
        )
        with torch.no_grad():
            losses = []
            for labels, images in loader:
                predictions = self.network.forward(
                    images.to(self.training_device)).cpu()
                loss = self.loss_function(predictions, labels)

                loss.backward()
                losses.append(loss.item())
            return numpy.mean(losses)

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

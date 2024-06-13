from abc import ABC, abstractmethod
from src.training.callbacks import base 

class BaseTrainer(base.TrainerCallbackMixin):
    """
    Base Training Pipeline
    for Deepfake Detection Classifier Model.
    """
    @abstractmethod
    def configure_optimizer(self):
        """
        Warning:  this is just empty shell for code implemented in other class.
        Configure optimization algorithm for training classifier network.
        """

    @abstractmethod
    def configure_device(self):
        """
        Warning: this is just empty shell for code implemented in other class.
        Configure and set device properties here.
        """

    @abstractmethod
    def load_model(self):
        """
        Warning: this is just empty shell for code implemented in other class.
        Configure and set network for training
        """

    @abstractmethod
    def load_metrics(self):
        """
        Warning: this is just empty shell for code implemented in other class.
        Configure evaluation metrics for model validation.
        """
    
    @abstractmethod
    def load_losses(self):
        """
        Warning: this is just empty shell for code implemented in other class.
        Configure loss functions for training network.
        """
    
    @abstractmethod
    def configure_lr_scheduler(self):
        """
        Warning: this is just empty shell for code implemented in other class.
        Configure LR Scheduler for adjusting learning rate during network training.
        """

    @abstractmethod
    def configure_seed(self):
        """
        Warning: this is just empty seed configure network on controller.
        Configure random seed for deterministic training process.
        """
    
    @abstractmethod
    def configure_callbacks(self):
        """
        Warning: this is just empty shell for code implemented in other classes.
        Configure callbacks to be executed during different events during training,
        validation or test.
        """
    @abstractmethod
    def train(self):
        """
        Warning: this is just empty shell for code implemented in other classes.
        Method for running full model training cycle.
        """
 
    @abstractmethod
    def inference(self):
        """
        Warning: this is just empty shell for code implemented in other classes.
        Method for generating predictions, used for model test and evaluation
        """
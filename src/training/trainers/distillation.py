from torch import nn 
import logging
import typing 
import os 
from tqdm import tqdm
from torch.utils import data
import torch
import numpy
import pathlib

from src.training.callbacks import (
    checkpoints,
    devices,
    base,
)
from src.training.trainers import base as base_trainer

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename="file_handler.log")
logger.addHandler(file_handler)

class DistillationTrainer(base_trainer.BaseTrainer):
    """
    Implementation of the training algorithm
    for distilling knowledge from one network to another.
    
    Parameters:
    ----------- 
        teacher_network - main complex architecture network
        student_network - secondary CNN network for compression
        optimizer - optimization algorithm for training.
        lr_scheduler - learning rate scheduling algorithm.
        train_device - device to use for training
        max_epochs - maximum number of training epochs.
        batch_size - size of the batch.
        temperature - softmax regulation temperature.
        alpha - alpha coefficient.
        log_dir - base log directory for storing information about
        checkpoints, cpu / gpu logs, losses, etc...
    """
    def __init__(self, 
        teacher_network: nn.Module, 
        student_network: nn.Module,
        optimizer: nn.Module, 
        student_loss_function: nn.Module,
        eval_metric: nn.Module,
        sim_loss: nn.Module,
        train_device: torch.DeviceObjType,
        max_epochs: int,
        batch_size: int,
        temperature: float,
        log_dir: pathlib.Path,
        alpha: float = 0.25,
        lr_scheduler: nn.Module = None,
        callbacks: typing.List[base.BaseCallback] = [],
        seed: int = None,
        reproducible: bool = False
    ):
        super(DistillationTrainer, self).__init__()

        self.teacher = teacher_network.to(train_device)
        self.student = student_network.to(train_device)

        self.student_loss_function: nn.Module = student_loss_function 
        self.eval_metric: nn.Module = eval_metric
        self.sim_loss: nn.Module = sim_loss

        self.optimizer = optimizer 
        self.lr_scheduler = lr_scheduler
        

        # reproducibility settings
        self.seed = seed 

        if reproducible == True:
            self.configure_reproducible()

        self.train_device = train_device

        if not isinstance(self.train_device, torch.DeviceObjType):
            self.train_device = train_device 

        self.batch_size: int = batch_size 
        self.max_epochs: int = max_epochs
        self.log_dir: pathlib.Path = log_dir
        self.temperature: float = temperature
        self.alpha: float = alpha
        self.callbacks = callbacks

        # initializing base callbacks in addition
        # to ones, specified in configuration.

        self.callbacks.extend(
            self.configure_callbacks()
        )

    def configure_loader(self, train_dataset: data.Dataset):
        return data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=max(0, os.cpu_count()-2),
            pin_memory=False
        )

    def configure_base_callbacks(self) -> typing.List[base.BaseCallback]:

        callbacks = []

        # setting up training device callback configuration

        if self.train_device.name == 'cpu':
            cpu_log_dir = os.path.join(self.log_dir, 'cpu_logs')
            callbacks.append(devices.CPUInferenceCallback(log_dir=cpu_log_dir))

        elif 'cuda' in self.train_device.name:
            gpu_log_dir = os.path.join(self.log_dir, 'gpu_cuda_logs')
            callbacks.append(devices.GPUInferenceCallback(log_dir=gpu_log_dir))

        checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        checkpoints.SnapshotCallback(
            snapshot_ext='.pt', 
            save_every=self.save_every, 
            log_dir=checkpoint_dir
        )

    def configure_reproducible(self):
        """
        Fixates parameters (randomness) of the experiment
        to enable further reproduciblity of the training.
        """
        torch.manual_seed(seed=self.seed)
        numpy.random.seed(seed=self.seed)

    def evaluate(self, validation_dataset: data.Dataset):

        validation_loader = self.configure_loader(validation_dataset)
        losses = []
        metrics = []

        with torch.zero_grad():
            self.on_validation_start()
            for images, labels in validation_loader:
                gpu_images = images.to(self.train_device)
                predictions = self.student.to(self.train_device).forward(gpu_images).cpu()
                val_loss = self.student_loss_function(predictions, labels)
                losses.append(val_loss.item())
        return (
            numpy.mean(losses), 
            numpy.mean(metrics)
        )

    def train(self, train_dataset: data.Dataset, validation_dataset: data.Dataset):

        self.student.train()
        self.teacher.train()

        self.on_init_start()

        train_loader = self.configure_loader(train_dataset)
        
        curr_loss = 0
        eval_metric = 0
        eval_loss = 0

        for epoch in range(self.max_epochs):
            epoch_losses = []

            self.on_train_epoch_start()

            for images, labels in tqdm(
                train_loader,
                desc='epoch: %s; train_loss: %s; val_loss: %s; metric: %s' % (
                    epoch, 
                    curr_loss, 
                    eval_loss,
                    eval_metric
                )
            ):
                self.on_train_batch_start()

                student_logits = self.student.to(
                self.train_device).forward(
                images.to(self.train_device)).cpu()

                teacher_logits = self.teacher.to(
                self.train_device).forward(
                images.to(self.train_device)).cpu()

                softmax_student_logs = student_logits / self.temperature
                softmax_teacher_logs = teacher_logits / self.temperature
        
                student_loss = self.student_loss_function(softmax_student_logs, labels)
                div_loss = self.sim_loss(softmax_teacher_logs, softmax_student_logs)
                
                total_loss = (0.75 * student_loss + self.alpha * div_loss)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                self.on_train_batch_end()

            self.on_train_epoch_end()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            curr_loss = numpy.mean(epoch_losses)
            eval_loss, eval_metric = self.evaluate(validation_dataset)
        
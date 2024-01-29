import torch
from torch import nn
import typing
import logging
from torch.utils import data
import numpy.random
import random
from src.training.trainers.regularization import EarlyStopping, LabelSmoothing
from tqdm import tqdm
import gc
from src.training.evaluators import sliced_evaluator
import os

trainer_logger = logging.getLogger("trainer_logger.log")
handler = logging.FileHandler("network_trainer_logs.log")
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
trainer_logger.setLevel(logging.WARN)
trainer_logger.addHandler(handler)


class NetworkTrainer(object):
    """
    Pipeline class, used for training 
    and evaluating neural networks

    Parameters:
    -----------

    network - (nn.Module) - neural network (nn.Module) object
    loss_function (nn.Module) - loss function to use for model training
    eval_metric - (nn.Module) - evaluation metric to use for model evaluation
    early_patience - indicates the number of tolerable epochs, when metric constantly decreases or not increasing.
    early_stop_dataset (data.Dataset) - dataset to use for Early Stopping.
    minimum_metric_difference - minimum difference between 2 eval metrics (Early Stopping parameter)
    batch_size - (int) - size of the data batch to pass inside the model
    max_epochs - (int) - maximum number of epochs to train
    optimizer - (nn.Module) - optimizer algorithm to use for training 
    lr_scheduler - (nn.Module) - Learning Rate Scheduling technique to use during training / fine-tuning
    train_device - (typing.Literal) - device to use for model training ("cuda:%s", "cuda", "cpu", "mps")
    loader_num_workers - number of CPU threads to use for data loading during training
    label_smoothing_eps - etta parameter for Label Smoothing regularization (default 0), which has no effect

    """

    def __init__(self,
                 network: nn.Module,
                 loss_function: nn.Module,
                 eval_metric: nn.Module,
                 early_patience: int,
                 early_stop_dataset: data.Dataset,
                 early_start: int,
                 minimum_metric_difference: int,
                 max_epochs: int,
                 batch_size: int,
                 optimizer: nn.Module,
                 checkpoint_dir: str,
                 output_weights_dir: str,
                 lr_scheduler: nn.Module = None,
                 train_device: typing.Literal['cpu', 'cuda', 'mps'] = 'cpu',
                 loader_num_workers: int = 1,
                 label_smoothing_eps: float = 0
                 ):

        self.network = network.to(train_device)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        self.train_device = train_device

        self.early_dataset = early_stop_dataset
        self.early_start = early_start
        self.early_stopper = EarlyStopping(
            patience=early_patience,
            min_diff=minimum_metric_difference
        )

        self.label_smoother = LabelSmoothing(etta=label_smoothing_eps)

        self.loss_function = loss_function
        self.eval_metric = eval_metric

        self.loader_num_workers = loader_num_workers
        self.seed_generator = torch.Generator()
        self.seed_generator.manual_seed(0)

        self.checkpoint_dir = checkpoint_dir
        self.output_weights_dir = output_weights_dir

    @staticmethod
    def seed_loader_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    def save_checkpoint(self,
                        loss: float,
                        epoch: int
                        ):
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "checkpoint_epoch_%s.pth" % str(epoch)
        )

        torch.save(
            {
                'anetwork': self.network.cpu().state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                'batch_size': self.batch_size,
                'loss': loss,
                'epoch': epoch,
                'train_device': self.train_device,
                'gpus_utilized': torch.cuda.device_count() if self.train_device == 'cuda' else 0,

                'gpu_devices': [] if self.train_device != 'cuda' else [
                    {
                        'gpu_name': torch.cuda.get_device_properties(device).name,
                        'total_memory': torch.cuda.get_device_properties(device).total_memory,
                        'number_of_multiprocessors': torch.cuda.get_device_properties(device).multi_processor_count
                    }
                    for device in range(torch.cuda.device_count())
                ]

            }, f=checkpoint_path
        )

    def train(self, train_dataset: data.Dataset):

        self.network.train()

        loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.loader_num_workers,
            worker_init_fn=self.seed_loader_worker,
            generator=self.seed_generator
        )

        best_loss = float('inf')
        best_eval_metric = 0
        loss_history = []

        for epoch in range(self.max_epochs):

            epoch_loss = 0

            for imgs, classes in tqdm(
                iterable=loader,
                desc='EPOCH %s, LOSS: %s, EVAL METRIC: %s ' % (
                    str(epoch), str(best_loss), str(best_eval_metric))):

                predictions = self.network.to(self.train_device).forward(
                    imgs.clone().detach().to(self.train_device))

                softmax_probs = torch.softmax(predictions, dim=1)
                smoothed_softmax_probs = self.label_smoother(softmax_probs)
                loss = self.loss_function(smoothed_softmax_probs, classes)
                epoch_loss += loss.item()

                # flushing cached predictions
                predictions.zero_()

                gc.collect()
                torch.cuda.empty_cache()

                loss.backward()
                self.optimizer.step()

            loss_history.append(epoch_loss)

            best_loss = min(best_loss, round(epoch_loss, 3))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.early_start <= (epoch + 1):
                metric = self.evaluate(self.early_dataset)
                best_eval_metric = max(metric, round(best_eval_metric, 3))
                step = self.early_stopper.step(metric)
                if step:
                    break

            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(epoch_loss, epoch)

        return best_loss, loss_history

    def evaluate(self, validation_dataset: data.Dataset, slicing=False):

        with torch.no_grad():

            if slicing:

                evaluator = sliced_evaluator.SlicedEvaluation(
                    network=self.network,
                    inf_device=self.train_device,
                    eval_metric=self.eval_metric
                )
                eval_metrics = evaluator.evaluate(self.early_dataset)
                return eval_metrics
            else:
                output_labels = torch.tensor([]).to(torch.uint8)
                try:
                    loader = data.DataLoader(
                        dataset=validation_dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.loader_num_workers,
                        worker_init_fn=self.seed_loader_worker,
                        generator=self.seed_generator,
                    )
                except (ValueError) as val_err:
                    trainer_logger.error(val_err)
                    raise SystemExit(
                        """RuntimeError: Validation dataset is empty,
                        therefore data loader could not load anything.
                        It can be due to invalid path to validation image dataset.
                        Make sure the path you provided under 'VAL_DATA_DIR' parameter 
                        is actually valid path and exists.""")

                for imgs, classes in tqdm(loader):

                    predictions = self.network.forward(
                        imgs.clone().detach().to(self.train_device)).cpu()

                    softmax_probs = torch.softmax(predictions, dim=1)
                    pred_labels = torch.argmax(
                        softmax_probs, dim=1, keepdim=False)
                    binary_labels = torch.where(pred_labels == classes, 1, 0)

                    predictions.zero_()
                    gc.collect()

                    output_labels = torch.cat([output_labels, binary_labels])

                # computing evaluation metric

                metric = self.eval_metric(
                    output_labels,
                    torch.ones_like(output_labels)
                )
                return metric

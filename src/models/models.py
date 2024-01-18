import torch
from torch import nn
import typing
import logging
from torch.utils import data
import numpy.random
import random
from regularization import EarlyStopping
import pathlib
from tqdm import tqdm
import gc
from src.evaluators import sliced_evaluator
from torch.utils.tensorboard.writer import SummaryWriter

trainer_logger = logging.getLogger("trainer_logger.log")
trainer_logger.setLevel("warning")

class NetworkPipeline(object):
    """
    Pipeline class, which encompasses
    model training / validation and inference
    processes.

    Parameters:
    -----------

    network - (nn.Module) -
    loss_function (nn.Module) -
    eval_metric - (nn.Module) -  
    early_patience - int - 
    early_stop_dataset (data.Dataset) -
    minimum_metric_difference - ()
    batch_size - (int) - 
    max_epochs - (int) -
    optimizer - (nn.Module) -
    major_version - (int)
    minor_version - (int)
    lr_scheduler - (nn.Module) - 
    train_device - (typing.Literal)
    loader_num_workers - ()

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
                 log_dir: str,
                 lr_scheduler: nn.Module = None,
                 train_device: typing.Literal['cpu', 'cuda', 'mps'] = 'cpu',
                 loader_num_workers: int = 1,
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
            patience=early_patience, min_diff=minimum_metric_difference)

        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        self.loader_num_workers = loader_num_workers
        self.seed_generator = torch.Generator()
        self.seed_generator.manual_seed(0)

    @staticmethod
    def seed_loader_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    def clean_up_checkpoints(self):
        """
        Function for cleaning up checkpoints directory
        """
        pathlib.Path("./checkpoints").rmdir()
        pathlib.Path("./checkpoints").mkdir(exist_ok=True)

    def save_checkpoint(self, 
        major_version: int,
        minor_version: int, 
        loss: float, 
        epoch: int
    ):

        # initializing checkpoints path, in case it is not initialized
        pathlib.Path("./checkpoints").mkdir(parents=True, exist_ok=True)

        # saving checkpoint

        torch.save(
            {
                'network': self.network.cpu().state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
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

            }, f='checkpoints/checkpoint_%s.%s.%s.pt' % (
                str(major_version),
                str(minor_version),
                str(epoch)
            )
        )

    def train(self, train_dataset: data.Dataset):

        self.network.train()

        loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_loader_worker,
            generator=self.seed_generator
        )

        best_loss = float('inf')
        loss_history = []

        for epoch in range(self.max_epochs):

            epoch_loss = 0

            for imgs, classes in tqdm(iterable=loader, desc='EPOCH %s' % str(epoch)):

                predictions = self.network.to(self.train_device).forward(
                    imgs.clone().detach().to(self.train_device))

                loss = self.loss_function(predictions.cpu(), classes)
                epoch_loss += loss.item()

                del predictions

                gc.collect()
                torch.cuda.empty_cache()

                loss.backward()
                self.optimizer.step()

            loss_history.append(epoch_loss)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.early_start <= (epoch + 1):
                metric = self.evaluate(self.early_dataset)
                print('evaluation metric: %s' % (str(metric)))
                step = self.early_stopper.step(metric)
                if step:
                    break

            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(epoch_loss, epoch)

            best_loss = min(best_loss, epoch_loss)
            print('epoch: %s; best_loss: %s' % (str(epoch + 1), best_loss))

        return best_loss, loss_history

    def evaluate(self, validation_dataset: data.Dataset, slicing=False):

        with torch.no_grad():

            if slicing:

                evaluator = sliced_evaluator.SlicedEvaluation(
                    network=self.network,
                    inf_device=self.train_device,
                    eval_metric=self.eval_metric,
                )
                eval_metrics = evaluator.evaluate(self.early_dataset)
                return eval_metrics
            else:
                output_labels = torch.tensor([]).to(torch.uint8)

                loader = data.DataLoader(
                    dataset=validation_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.loader_num_workers,
                    worker_init_fn=self.seed_loader_worker,
                    generator=self.seed_generator,
                )

                for imgs, classes in tqdm(loader):

                    predictions = self.network.forward(
                        imgs.clone().detach().to(self.train_device)).cpu()

                    classes = torch.as_tensor(classes)
                    predicted_labels = torch.as_tensor([
                        torch.argmax(predictions[idx], axis=0)
                        for idx in range(len(predictions))
                    ])

                    del predictions
                    gc.collect()

                    binary_labels = (
                        classes == predicted_labels).to(torch.int8)
                    output_labels = torch.cat([output_labels, binary_labels])

                # computing evaluation metric

                metric = self.eval_metric(
                    output_labels,
                    torch.ones_like(output_labels)
                )
                return metric

    def predict(self, input_images: typing.List[str]):
        self.network.eval()
        batch_imgs = torch.stack(input_images).to(self.train_device)
        predictions = self.network.forward(batch_imgs).cpu()
        predicted_labels = [
            torch.argmax(predictions[idx])
            for idx in range(len(predicted_labels))
        ]
        return predicted_labels

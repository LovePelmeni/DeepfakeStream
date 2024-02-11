import torch
from torch import nn
import typing
import logging
from torch.utils import data
import numpy.random
import random
from tqdm import tqdm
import gc
import os

from src.training.regularizations.regularization import EarlyStopping, LabelSmoothing
from src.training.evaluators import sliced_evaluator
from torch.utils.tensorboard.writer import SummaryWriter

trainer_logger = logging.getLogger("trainer_logger.log")
handler = logging.FileHandler("network_trainer_logs.log")
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
trainer_logger.setLevel(logging.WARN)
trainer_logger.addHandler(handler)


from abc import ABC, abstractmethod
import os
class BaseTrainer(ABC):

    @abstractmethod
    def load_snapshot(self, snapshot_path: str) -> None:
        """
        Loads state of the model from snapshot path
        Parameters:
        -----------
            snapshot_path (str) - path to the snapshot
        """

    @abstractmethod
    def save_snapshot(self, **kwargs) -> None:
        """
        Saves checkpoint (snapshot) of the model
        under specific file path and format.

        Possible formats:
            - "pth", "pt", "onnx"

        Example arguments:
            path = "path/to/save/snapshot/"
            format = ".onnx"
            output_path = os.path.join(path, "model" + format)
            model.save(output_path)
        """
    
    @abstractmethod
    def prepare_loader(self, dataset: data.Dataset) -> data.DataLoader:
        """
        Prepares loader for training model
        """

    @abstractmethod
    def train(self, dataset: data.Dataset) -> float:
        """
        Trains classifier with 'dataset'
        Parameters:
        -----------
            dataset - (data.Dataset) - training dataset
        Returns:
            - train loss over epochs (float number)
        """
class NetworkTrainer(BaseTrainer):
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
                 log_dir: str,
                 save_every: int,
                 lr_scheduler: nn.Module = None,
                 train_device: typing.Literal['cpu', 'cuda', 'mps'] = 'cpu',
                 loader_num_workers: int = 1,
                 label_smoothing_eps: float = 0,
                 distributed: bool = False,
                 reproducible: bool = False,
                 seed: int = None
                 ):

        if distributed:

            device_ids = [
                int(cuda_id) for cuda_id 
                in train_device.split(":")[-1].split(",")
            ]

            if not len(device_ids): 
                raise ValueError("""invalid device name 
                for distributed training, shoule be in a format: 'cuda:x,y,z', 
                however, got: '%s'""" % train_device)

            self.network = nn.parallel.DistributedDataParallel(
                module=network, 
                device_ids=device_ids
            )

        self.reproducible = reproducible
        self.seed = seed
        self.distributed = distributed

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
        self.save_every = save_every

        self.checkpoint_dir = checkpoint_dir
        self.output_weights_dir = output_weights_dir

        self.overall_writer = SummaryWriter(
            log_dir=log_dir, 
            max_queue=10
        )

        self.encoder_writer = SummaryWriter(
            log_dir=os.path.join(log_dir, "encoder"), 
            max_queue=10
        )

        self.custom_net_writer = SummaryWriter(
            log_dir=os.path.join(log_dir, "custom_network"), 
            max_queue=10
        )
    
    def track_network_params(self, writer: SummaryWriter, global_step: int):
        """
        Method for tracking 
        network weight distribution

        Parameters:
        ----------
            global_step - (int) - number of batches run previously
        """
        for param_name, param in self.network.named_parameters():
            
            if 'weight' in param_name:
                tag_name = 'weights'

            elif 'bias' in param_name:
                tag_name = 'biases'

            writer.add_histogram(
                tag="%s/%s" % (param_name, tag_name),
                values=param.clone().cpu().data.numpy(),
                global_step=global_step
            )
        
    @staticmethod
    def seed_loader_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    def save_snapshot(self,
                        loss: float,
                        epoch: int
                        ):
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "checkpoint_epoch_%s.pth" % str(epoch)
        )

        snapshot = {
                'network_state': self.network.cpu().state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'lr_scheduler_state': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
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

        }
        if self.lr_scheduler is not None:
            snapshot['lr_scheduler_state'] = self.lr_scheduler.state_dict()

        torch.save(snapshot, f=checkpoint_path)

    def load_snapshot(self, snapshot_path: str) -> None:
        """
        Loads the network from the 'snapshot_path'
        directory.
        """
        if os.path.exists(snapshot_path):
            file_format = os.splitext(os.basename(snapshot_path))[1]

            if file_format.lower() in ("pt", "pth"):
                config = torch.load(snapshot_path)

                self.network = self.network.load_state_dict(config['model_state'])
                self.network = self.network.to(config['train_device'])

    def get_reproducible_loader(self, worker_seed: int, loader: data.DataLoader) -> data.DataLoader:
        """
        Sets the deterministic behaviour
        of dataloader's workers
        """
        if not self.seed:
            raise ValueError(msg="""you did not specified any seed. 
            Pass `seed=value` argument to the object 
            for setting up the seed for reproducibility""")

        seed_generator = torch.Generator(device=self.train_device)
        seed_generator.manual_seed(seed=worker_seed)
        loader.worker_init_fn = self.seed_loader_worker
        loader.generator = seed_generator
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
 
    def reset_reproducible(self, loader: data.DataLoader):
        loader.worker_init_fn = None
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        loader.generator = None
        
    def prepare_loader(self, dataset: data.Dataset) -> data.DataLoader:
        """
        Prepares dataset for training
        Options:
            - in case 'distributed' set to False,
            returns standard data loader 
            - otherwise, returns data loader,
            adapted for training on multiple GPU devices
        Returns:
            - data.DataLoader object
        """
        if self.distributed:
            loader = data.DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.loader_num_workers,
                sampler=data.DistributedSampler(dataset=dataset)
            )
        else:
            loader = data.DataLoader(
                dataset=dataset,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.loader_num_workers,
                pin_memory=False
            )

        if self.reproducible:
            return self.get_reproducible_loader(
                worker_seed=self.seed, 
                loader=loader
            )
        else:
            return self.get_reproducible_loader(
                worker_seed=self.seed,
                loader=loader
            )

    def train(self, train_dataset: data.Dataset):

        self.network.train()

        loader = self.prepare_loader(dataset=train_dataset)
        global_step = 0
        best_loss = 0

        for epoch in range(self.max_epochs):

            epoch_loss = 0

            for imgs, classes in tqdm(
                iterable=loader,
                desc='EPOCH %s, LOSS: %s, EVAL METRIC: %s ' % (
                    str(epoch), str(best_loss), str(best_eval_metric))):

                predictions = self.network.forward(
                    imgs.clone().detach().to(self.train_device))

                softmax_probs = torch.softmax(predictions, dim=1)

                # computing one hot distribution
                one_hots = torch.zeros_like(softmax_probs)
                
                for idx, class_ in enumerate(classes):
                    one_hots[idx][class_] = 1

                smoothed_one_hots = torch.as_tensor(self.label_smoother(one_hots))

                loss = self.loss_function(softmax_probs, smoothed_one_hots)
                
                epoch_loss += loss.item()

                # flushing cached predictions
                predictions.zero_()

                # clearing up dynamic memory
                gc.collect()
                torch.cuda.empty_cache()

                loss.backward()
                self.optimizer.step()

                # tracking distributions of network parameters
                self.track_network_params(
                    writer=self.encoder_writer, 
                    global_step=global_step
                )

                self.track_network_params(
                    writer=self.custom_net_writer, 
                    global_step=global_step
                )

                global_step += 1

            # tracking training loss history

            self.overall_writer.add_scalar(
                tag='training loss',
                scalar_value=numpy.mean(epoch_loss)
            )

            best_loss = min(best_loss, numpy.mean(best_loss))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.early_start <= (epoch + 1):

                metric = self.evaluate(self.early_dataset)
                best_eval_metric = max(metric, round(best_eval_metric, 3))
                step = self.early_stopper.step(metric)

                # tracking evaluation metric
                self.overall_writer.add_scalar(
                    tag='evaluation metric', 
                    scalar_value=best_eval_metric,
                    global_step=epoch
                )

                if step:
                    break

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch_loss, epoch)

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
                    loader = self.prepare_loader(dataset=validation_dataset)
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
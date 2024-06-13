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
from torch.utils.tensorboard.writer import SummaryWriter
from src.training import regularizations as regularization
from src.training.evaluators import sliced_evaluator
import pathlib

trainer_logger = logging.getLogger("trainer_logger.log")
handler = logging.FileHandler("network_trainer_logs.log")
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
trainer_logger.setLevel(logging.WARN)
trainer_logger.addHandler(handler)

torch.autograd.set_detect_anomaly(True)


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
                 log_dir: str,
                 save_every: int,
                 scheduler: nn.Module = None,
                 train_device: typing.Literal['cpu', 'cuda', 'mps'] = 'cpu',
                 loader_num_workers: int = 1,
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

        self.lr_scheduler = scheduler
        self.train_device = train_device

        self.early_dataset = early_stop_dataset
        self.early_start = early_start
        self.early_stopper = regularization.EarlyStopping(
            patience=early_patience,
            min_diff=minimum_metric_difference
        )

        self.loss_function = loss_function
        self.eval_metric = eval_metric

        self.loader_num_workers = loader_num_workers
        self.save_every = save_every

        self.log_dir = pathlib.Path(log_dir)
        self.device_utilization_dir = pathlib.Path(
            os.path.join(
                log_dir, 
                "device_utilization"
            )
        )
        self.checkpoint_dir = pathlib.Path(
            os.path.join(
                log_dir, 
                "checkpoints"
            )
        )
        self.output_weights_dir = pathlib.Path(
            os.path.join(
                log_dir, 
                "weights"
            )
        )

        self.network_writer = SummaryWriter(  # logging dir should preferably contain (name of the experiment,version,timestamp)
            log_dir=os.path.join(log_dir, "network_params"),
            max_queue=10
        )

        # creating directories, in case some of them does not exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_weights_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.device_utilization_dir, exist_ok=True)

    def freeze_layers(self, layers):
        """
        Disables gradient flow 
        from specific network layers.

        Parameters:
        -----------
            layers - network layers to freeze
        """
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_layers(self):
        """
        Unfreezes all layers of the network,
        in case some of them was freezed.
        """
        for param in self.network.parameters():
            param.requires_grad = True

    def save_model(self,
                   filename: str,
                   test_input: torch.Tensor,
                   format: typing.Literal['onnx', 'pth', 'pt']
                   ):
        """
        Saves network under specific format
        and filename

        NOTE:
            you should provide format without extension "." 
            Example:
                onnx, pth or pt 

            Not:
                .onnx, .pth or .pt
        """
        model_path = os.path.join(
            self.output_weights_dir,
            filename + ".%s" % format
        )

        if format == 'onnx':
            torch.onnx.format(self.network, test_input, model_path)

        elif format in ('pt', 'pth'):
            torch.jit.save(m=self.network, f=model_path)
        else:
            options = ["onnx", "pth", "pt"]
            raise ValueError(
                msg='invalid model saving format provided. Available options: %s' % options)

    def track_network_params(self, writer: SummaryWriter, global_step: int):
        """
        Method for tracking 
        network weight distribution

        Parameters:
        ----------
            global_step - (int) - number of batches run previously
        """
        for param_name, param in self.network.named_parameters():

            if (param.requires_grad == True):

                if ('weight' in param_name):
                    tag_name = 'weights'

                elif ('bias' in param_name):
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

                self.network = self.network.load_state_dict(
                    config['model_state'])
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
        seed_generator.manual_seed()
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
            return loader

    def train(self, train_dataset: data.Dataset):

        self.network.train()

        loader = self.prepare_loader(dataset=train_dataset)
        global_step = 0
        best_loss = float('inf')
        current_loss = float('inf')
        best_eval_metric = 0

        for epoch in range(self.max_epochs):

            epoch_loss = []

            for imgs, classes in tqdm(
                iterable=loader,
                desc='EPOCH %s, BEST_LOSS: %s, CURR_LOSS: %s, EVAL METRIC: %s ' % (
                    (epoch+1), best_loss, current_loss, best_eval_metric)):

                gpu_imgs = imgs.to(self.train_device)
                raw_logits = self.network.forward(gpu_imgs)

                loss = self.loss_function(raw_logits, classes)

                epoch_loss.append(loss.detach().cpu().item())

                loss.backward()
                self.optimizer.step()

                # tracking distributions of network parameters
                global_step += 1

            # tracking training loss history

            best_loss = min(best_loss, numpy.mean(epoch_loss))
            current_loss = numpy.mean(epoch_loss)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.early_start <= (epoch + 1):

                metric = self.evaluate(self.early_dataset)
                best_eval_metric = max(metric, round(best_eval_metric, 3))
                step = self.early_stopper.step(metric)

                if step:
                    break

            if (epoch + 1) % self.save_every == 0:
                self.save_snapshot(epoch_loss, epoch)

            # saving tracked information to tensorboard logs directory
            self.track_network_params(
                writer=self.network_writer,
                global_step=global_step
            )
        return best_loss

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
                output_labels = numpy.array([]).astype(numpy.uint8)
                output_classes = numpy.array([]).astype(numpy.uint8)
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

                    probs = self.network.forward(
                        imgs.clone().detach().to(self.train_device))

                    cpu_probs = probs.cpu()

                    pred_labels = torch.argmax(
                        cpu_probs, dim=1, keepdim=False)

                    output_labels = numpy.concatenate(
                        [output_labels, pred_labels])
                    output_classes = numpy.concatenate(
                        [output_classes, classes])

                    cpu_probs.zero_()
                    gc.collect()

                # computing evaluation metric

                metric = self.eval_metric(
                    output_labels,
                    output_classes
                )
                return metric




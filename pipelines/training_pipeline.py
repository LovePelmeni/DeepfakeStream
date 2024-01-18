import argparse
import torch.nn 
import torch 
import os
import pathlib
import numpy 
import random

from torch.utils.tensorboard.writer import SummaryWriter
from pipelines import train_utils as utils

from src.models import models 
from src.losses import losses 
from src.metrics import metrics 
from src.datasets import datasets
from src.augmentations import augmentations


def training_pipeline():
    """
    Training pipeline 
    for running experiments
    """
    parser = argparse.ArgumentParser(description="Training Pipeline")
    arg = parser.add_argument 

    # data and output settings
    arg("--train-data-path", type=str, required=True, dest='train_data_path', help='path to the training data')
    arg("--val-data-path", type=str, required=True, dest='val_data_path', help='path to the validation data')
    arg("--output-dir", type=str, default='weights/', dest='output_dir', help='path for storing network weights')
    arg("--checkpoint-dir", type=str, default='checkpoints/', dest='checkpoint_dir', help='path for storing training checkpoints')
    arg('--config-dir', type=str, dest='config', help='path to .json data configuration file.')
    arg("--log-dir", type=str, default='logs/', dest='log_dir', help='directory for storing logs')
    arg("--data-size-identical", type=bool, default=False, dest='size_identical', help='In case all image data have the same size. It turns on optimization process, which speeds up training drastically')

    # hardware settings
    arg("--use-cuda", type=bool, default=False, dest='use_cuda', help='presence of cuda during training')
    arg("--use-cpu", type=bool, default=True, dest='use_cpu', help='presence of CPU during training')
    arg("--num-workers", type=int, default=2, dest='num_workers', help='number of CPU threads to be involved during training')
    arg("--gpu-id", type=str, default='', help='ID of GPU to use for training')

    # training-specific settings

    arg("--classifier-name", type=str, dest='classifier_name', help='full name of classifier model, you want to usef or training.')
    arg("--optimizer", type=str, default='adam', help='name of the optimizer to use.')
    arg("--lr-scheduler", type=str, default='', help='LR Scheduling technique, in format: "<name>[param1:<>,param2:<>,param3:<>]"')
    arg("--loss-fn", type=str, dest='loss_fn', help='loss function for training classifier')
    arg("--eval-metric", type=str, dest='eval_metric', help='evaluation metric for validating classifier')
    arg("--max-epochs", type=int, dest='max_epochs', required=True, help='number of training epochs')
    arg("--batch-size", type=int, dest='batch_size', default=32, help='size of the batch for training')

    # regularization techniques

    arg("--label-smoothing", type=bool, default=False, help='apply label smoothing')    
    arg("--loss-weights", type=str, default='', help='loss weights for the classes')
    arg("--weight-decay", type=float, default=1, help='constant for decaying weights during each training epoch, ')
    arg("--loss-reduction", type=str, default='mean', choices=["mean", "sum", "none"], help='reduction technique for loss function')

    arg("--early-stopping", type=bool, default=True, help='use early stopping during training or not. Default: True')
    arg("--early-stopping-patience", default=5, help='number of bad epochs to wait, before shutting down training')
    arg("--early-stopping-metric-diff", default=0.7, help='minimum difference between epoch metrics')
    arg("--early-stopping-data-prop", type=float, default=0.2, help='proportion of validation data to use for early stopping')

    arg("--seed", type=int, default=42, help='seed for fixating generation of random entities during training')
    
    # experiment settings (for experiment tracking)

    arg("--prefix", type=str, dest='exp_prefix', help='unique prefix of the experiment. Example: 45g2srgw756')

    # parsing arguments

    args = parser.parse_args()


    # Setting reproducability
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # setting up data and output settings

    train_dir = pathlib.Path(args.train_data_path)
    valid_dir = pathlib.Path(args.val_data_path)
    output_dir = pathlib.Path(args.output_dir)
    log_dir = pathlib.Path(args.log_dir + "/" + "exp_%s" % args.exp_prefix)
    config_dir = pathlib.Path(args.config_dir)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)

    log_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    if (args.size_identical == True):
        torch.backends.cudnn.benchmark = True

    data_config = utils.load_config(config_dir) # data configuration (dict-like object)

    train_image_paths, train_image_labels = utils.get_train_data(train_dir)

    train_augmentations = augmentations.get_training_augmentations(
        HEIGHT=data_config['image_height'], 
        WIDTH=data_config['image_width']
    )
    
    val_image_paths, val_image_labels = utils.get_val_data(valid_dir)

    val_augmentations = augmentations.get_validation_augmentations(
        HEIGHT=data_config['image_height'], 
        WIDTH=data_config['image_width']
    )

    train_dataset = datasets.DeepfakeDataset(
        image_paths=train_image_paths,
        image_labels=train_image_labels,
        augmentations=train_augmentations,
    )

    validation_dataset = datasets.DeepfakeDataset(
        image_paths=val_image_paths,
        image_labels=val_image_labels,
        augmentations=val_augmentations
    )

    # setting up snapshot summarty writer 
    log_dir + "/" + args.exp_prefix
    summary_writer = SummaryWriter(log_dir=log_dir)

    # setting up hardware settings

    device_name = "cpu"

    if args.use_cuda:

        device_name = "cuda" if args.use_cuda else "cpu"

        if len(args.gpu_id) != 0:

            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            if isinstance(args.gpu_id, str) and args.gpu_id[0].isdigit():
                device_name = "cuda:%s" % (args.gpu_id[0])

    train_device = torch.device(device_name)
    num_cpu_workers = int(args.num_workers)

    # training-specific settings

    network = utils.load_network(args.classifier_name)
    optimizer = utils.get_optimizer(args.optimizer)
    lr_scheduler = utils.get_lr_scheduler(args.lr_scheduler)

    loss_weights = torch.as_tensor(args.loss_weights.split(",")).to(torch.float16)
    loss_function = losses.WeightedLoss(name=args.loss_fn, weights=loss_weights)
    evaluation_metric = utils.get_evaluation_metric(args.eval_metric)

    batch_size = int(args.batch_size)
    max_epochs = int(args.max_epochs)
    
    # setting up classifier training pipeline 

    trainer = models.NetworkPipeline(
        network=network,
        loss_function=loss_function,
        eval_metric=evaluation_metric,
        train_device=train_device,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr_scheduler=lr_scheduler,
        checkpoint_dir=checkpoint_dir,
        output_weights_dir=output_dir,
        summary_writer=summary_writer,
        loader_num_workers=num_cpu_workers
    )

    # training classifier
    _, _ = trainer.train(train_dataset=train_dataset)


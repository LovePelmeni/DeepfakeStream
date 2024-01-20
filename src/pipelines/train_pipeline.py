import argparse
import torch.nn 
import torch 
import os
import pathlib
import numpy 
import random
import matplotlib.pyplot as plt
import logging

from torch.utils.tensorboard.writer import SummaryWriter
import train_utils as utils

from src.models import models 
from src.losses import losses
from src.datasets import datasets
from src.augmentations import augmentations


info_logger = logging.getLogger("pipeline_info_logger")
err_logger = logging.getLogger("pipeline_error_logger")

error_handler = logging.FileHandler(filename="pipeline_issue_logs.log")
info_handler = logging.FileHandler(filename="pipeline_debug_logs.log")

# setting up handler boundary level

error_handler.setLevel(level='warning')
info_handler.setLevel(level='info')

# adding handlers

err_logger.addHandler(hdlr=error_handler)
info_logger.addHandler(hdlr=info_handler)

"""
Pipeline for training networks. 
Based on experiment configuration.
Offers CLI for manual training locally,
or can be executed via specifying '.env' file
and running corresponding 'train_pipeline.sh' script.
"""

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
    arg("--labels-path", type=str, required=True, dest='labels_path', help='path to CSV / JSON file, containing labels for training and validation data')
    arg("--output-path", type=str, default='weights/', dest='output_dir', help='path for storing network weights')
    arg("--checkpoint-path", type=str, default='checkpoints/', dest='checkpoint_dir', help='path for storing training checkpoints')
    arg('--config-path', type=str, dest='config', help='path to .json data configuration file.')
    arg("--log-path", type=str, default='logs/', dest='log_dir', help='directory for storing logs')
    # hardware settings
    arg("--use-cuda", type=bool, default=False, dest='use_cuda', help='presence of cuda during training')
    arg("--use-cpu", type=bool, default=True, dest='use_cpu', help='presence of CPU during training')
    arg("--num-workers", type=int, default=3, dest='num_workers', help='number of CPU threads to be involved during training')
    arg("--gpu-id", type=str, default='', required=False, help='ID of GPU to use for training')
    arg("--use-cudnn-bench", type=str, dest='cudnn_bench', default=False, help='In case your data has the same shape, turning this option to "True" can dramatically speed up on training')

    # reproducability settings

    arg("--seed", type=int, default=42, help='seed for fixating generation of random entities during training')
    
    # experiment settings (for experiment tracking)

    arg("--prefix", type=str, dest='exp_prefix', help='unique prefix of the experiment. Example: 45g2srgw756')
    arg("-log-execution", default=True, help='Option for use / disabling logging during pipeline execution.')
    
    # parsing arguments

    args = parser.parse_args()

    if (args.log_execution == False):

        err_logger.disabled = True
        info_logger.disabled = True
    else:
        err_logger.disabled = False 
        info_logger.disabled = False 

    info_logger.info("configurations parsed successfully")

    # Setting reproducability
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # setting up data and output settings

    train_dir = pathlib.Path(args.train_data_path)
    valid_dir = pathlib.Path(args.val_data_path)
    labels_dir = pathlib.Path(args.labels_path)
    output_dir = pathlib.Path(args.output_dir)
    log_dir = pathlib.Path(args.log_dir + "/" + "exp_%s" % args.exp_prefix)
    config_dir = pathlib.Path(args.config_dir)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)


    log_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    info_logger.info("recreated paths")

    torch.backends.cudnn.benchmark = args.cudnn_bench

    if (args.cudnn_bench == True):
        err_logger.warn(
            "cunn.benchmark mode has been turned on. \
            Make sure the data has the same size.") 

    # loading experiment configuration

    try:
        exp_config = utils.load_config(config_dir) # data configuration (dict-like object)
        labels_file = utils.load_labels_to_csv(labels_dir) # loading data annotations
        image_names = labels_file['videoName'].apply(lambda path: path.split("/")[-1])

        # loading training data
        train_image_paths = utils.load_images(train_dir)

        train_image_labels = image_names[
            numpy.where(
                numpy.char.endswith(
                    train_image_paths, image_names
                )
            )[0]
        ]

        train_augmentations = augmentations.get_training_augmentations(
            HEIGHT=exp_config['data']['height'], 
            WIDTH=exp_config['data']['width']
        )

        # loading validation data
        val_image_paths = utils.load_images(valid_dir)

        val_image_labels = image_names[
            numpy.where(
                numpy.char.endswith(
                    train_image_paths, image_names
                )
            )[0]
        ]

        val_augmentations = augmentations.get_validation_augmentations(
            HEIGHT=exp_config['data']['height'], 
            WIDTH=exp_config['data']['width']
        )
    except(Exception) as config_err:
        err_logger.error(config_err)

    try:
        # setting up training dataset for training
        train_dataset = datasets.DeepfakeDataset(
            image_paths=train_image_paths,
            image_labels=train_image_labels,
            augmentations=train_augmentations,
        )
        # setting up validation dataset for evaluation
        validation_dataset = datasets.DeepfakeDataset(
            image_paths=val_image_paths,
            image_labels=val_image_labels,
            augmentations=val_augmentations
        )

    except(Exception) as init_err:
        err_logger.error(init_err)

    # setting up snapshot summarty writer
    summary_writer = SummaryWriter(log_dir=log_dir)

    # setting up hardware settings

    device_name = "cpu"
    
    if args.use_cuda and args.use_cpu:
        raise RuntimeError("You cannot use CPU and GPU at the same time. \
        Turn one of the options to False")

    if args.use_cuda:

        device_name = "cuda" if args.use_cuda else "cpu"

        if len(args.gpu_id) != 0:

            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            if isinstance(args.gpu_id, str) and args.gpu_id[0].isdigit():
                device_name = "cuda:%s" % (args.gpu_id[0])
        else:
            err_logger.warn("you didnt provide any GPU ids. \
            Cuda is going to leverage all available GPUs during training.")

    train_device = torch.device(device_name)
    num_cpu_workers = int(args.num_workers)

    if num_cpu_workers < 3:
        err_logger.warn("You've set number of cpu threads to be less, \
        than the baseline (recommended is 3).")

    elif num_cpu_workers > 5:
        err_logger.warn("You've set number of cpu threads to be greater, \
         than it is recommended. (5 threads is usually considered as enough'")

    # training-specific settings

    network = utils.load_network(exp_config['network'])
    optimizer = utils.get_optimizer(exp_config['optimizer'], model=network)
    lr_scheduler = utils.get_lr_scheduler(exp_config['scheduler'], optimizer) if 'scheduler' in exp_config else None
    loss = utils.get_loss(exp_config['loss']['name'])

    loss_function = losses.WeightedLoss(
        loss_function=loss, 
        weights=exp_config['loss']['weights'],
        weight_type=exp_config['loss']['type']
    )

    evaluation_metric = utils.get_evaluation_metric(exp_config['eval_metric'])

    batch_size = int(exp_config['batch_size'])
    max_epochs = int(exp_config['max_epochs'])
    
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
        loader_num_workers=num_cpu_workers
    )

    # training classifier
    training_loss, loss_history = trainer.train(train_dataset=train_dataset)

    # plot of the loss function change during training

    loss_figure = plt.figure(figsize=(30, 30))
    _, ax = plt.plot(loss_history)

    ax.axes.set_xticklabels(
        labels=[
            'epoch %s' % str(epoch) 
            for epoch in 
            range(len(loss_history))
        ]
    )

    summary_writer.add_figure(tag='train loss history', figure=loss_figure)
    summary_writer.add_scalar(tag='training loss', scalar_value=training_loss)

    # evaluating classifier on entire validation dataset
    evaluation_metric = trainer.evaluate(
        validation_dataset=validation_dataset
    )
    # saving evaluation metric results to summary writer
    summary_writer.add_scalar(
        "evaluation metric: ", 
        scalar_value=evaluation_metric, 
        double_precision=True
    )
    print('training completed.')

training_pipeline()



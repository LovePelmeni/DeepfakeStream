import argparse
import torch.nn
import torch
import os
import pathlib
import numpy
import random
import matplotlib.pyplot as plt
import logging
import sys

from torch.utils.tensorboard.writer import SummaryWriter
import src.pipelines.train_utils as utils

from src.training.trainers import models


info_logger = logging.getLogger("train_pipeline_info_logger")
err_logger = logging.getLogger("train_pipeline_error_logger")
runtime_logger = logging.getLogger("train_pipeline_runtime_logger")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

error_handler = logging.FileHandler(filename="train_pipeline_issue_logs.log")
info_handler = logging.FileHandler(filename="train_pipeline_info_logs.log")
runtime_handler = logging.StreamHandler(stream=sys.stdout)

# setting up logging formatters

error_handler.setFormatter(formatter)
info_handler.setFormatter(formatter)
runtime_handler.setFormatter(formatter)

# setting up handler boundary level

error_handler.setLevel(level=logging.WARNING)
info_handler.setLevel(level=logging.INFO)
runtime_logger.setLevel(level=logging.DEBUG)

# adding handlers

err_logger.addHandler(hdlr=error_handler)
info_logger.addHandler(hdlr=info_handler)
runtime_logger.addHandler(hdlr=runtime_handler)

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
    parser = argparse.ArgumentParser(description="CLI-based Training Pipeline")
    arg = parser.add_argument

    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument(
        "-h", "--help", 
        action='help', 
        help='helper method for printing information about available commands'
    )

    # data and output settings
    required_settings = parser.add_argument_group("Required training settings")

    required_settings.add_argument("--train-data-path", type=str, required=True,
        dest='train_data_path', help='path to the training data')
    required_settings.add_argument("--train-labels-path", type=str, required=True,
        dest='train_labels', help='.csv file labels for training data.')
    required_settings.add_argument("--val-data-path", type=str, required=True,
        dest='val_data_path', help='path to the validation data')
    required_settings.add_argument("--val-labels-path", type=str, required=True,
        dest='val_labels', help='.csv file labels for validation data')
    required_settings.add_argument("--output-path", type=str, default='weights/',
        dest='output_dir', help='path for storing network weights')
    required_settings.add_argument("--checkpoint-path", type=str, default='checkpoints/',
        dest='checkpoint_dir', help='path for storing training checkpoints')
    required_settings.add_argument('--config-path', type=str, dest='config_dir',
        help='path to .json train configuration file, containing information for training pipeline.')
    required_settings.add_argument("--log-path", type=str, default='logs/',
        dest='log_dir', help='directory for storing logs')

    # hardware settings
    train_device_settings = parser.add_mutually_exclusive_group("Hardware device selection: ")

    train_device_settings.add_argument("--use-cuda", type=bool, dest='use_cuda',
        help='use CUDA backend for model training', action='store_true', required=True)

    train_device_settings.add_argument("--use-mps", type=bool, action='store_true', dest="use_mps",
        help="use MAC mps backend for training", required=True)

    train_device_settings.add_argument("--use-cpu", type=bool, action='store_true', dest='use_cpu',
        help='use CPU backend for training')

    # additional hardware related settings
    optional_hard_settings = parser.add_argument_group("Optional hardware training settings")

    optional_hard_settings.add_argument("--cpu-num-workers", type=int, default=3, dest='num_workers',
        help='number of CPU threads for loading dataset during training')

    optional_hard_settings.add_argument("--gpu-ids", type=str, default='', required=False,
        help='ID of GPU to use for training')

    optional_hard_settings.add_argument("--use-cudnn-bench", type=bool, dest='cudnn_bench', default=False,
        help='In case your data has the same shape, turning this option to "True" can dramatically speed up on training')

    # reproducability settings
    reprod_settings = parser.add_argument_group("Reproducibility settings")

    reprod_settings.add_argument("--seed", type=int, default=42,
        help='seed for fixating generation of random entities during training')

    # experiment settings (for experiment tracking)

    reprod_settings.add_argument("--log-execution", default=True,
        help='Option for use / disabling logging during pipeline execution.')

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

    train_dataset_dir = pathlib.Path(args.train_data_path)
    train_labels_dir = pathlib.Path(args.train_labels)
    valid_dataset_dir = pathlib.Path(args.val_data_path)
    val_labels_dir = pathlib.Path(args.val_labels)
    output_dir = pathlib.Path(args.output_dir)
    log_dir = pathlib.Path(args.log_dir)
    config_dir = pathlib.Path(args.config_dir)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    info_logger.info("recreated paths")
    torch.backends.cudnn.benchmark = args.cudnn_bench

    if (args.cudnn_bench == True):
        err_logger.warning(
            """cunn.benchmark mode has been turned on.
            Make sure the data has the same size.""")

    # loading experiment configuration

    try:
        # data configuration (dict-like object)
        exp_config = utils.load_config(config_dir)

        # loading training data
        train_image_paths = numpy.asarray(utils.load_image_paths(train_dataset_dir))
        train_labels = utils.load_labels_to_csv(train_labels_dir)['class']

        # loading validation data
        val_image_paths = numpy.asarray(utils.load_image_paths(valid_dataset_dir))
        val_labels = utils.load_labels_to_csv(val_labels_dir)['class']
        

    except(FileNotFoundError) as err:
        raise err

    except (Exception) as config_err:
        err_logger.error(config_err)
        raise RuntimeError("failed to load image and label data.")

    try:
        # setting up training dataset for training
        train_dataset = utils.load_deepfake_dataset(train_image_paths, train_labels)
    
        # setting up validation dataset for evaluation
        validation_dataset = utils.load_deepfake_dataset(val_image_paths, val_labels)

    except (Exception) as init_err:
        err_logger.error(init_err)
        raise RuntimeError("Failed to initialize DeepFake datasets.")


    # loading early stopping configuration, in case it is presented

    if 'early_stopping' in exp_config:
      
        early_patience = exp_config.get("patience", 2)
        early_start = exp_config.get("start", 3)
        min_metric_difference = exp_config.get('min_diff', 0.01) # minimum difference between epoch metrics

        # for early stopping, we usually pick 15% of the validation dataset

        early_indices = numpy.random.choice(
            a=numpy.arange(len(validation_dataset)), 
            size=max(int(len(validation_dataset) * 0.15), 1)
        )

        early_image_paths = val_image_paths[early_indices]
        early_image_labels = val_labels[early_indices]

        early_dataset = utils.load_deepfake_dataset(
            early_image_paths, 
            early_image_labels
        )
        
    else:
        early_patience = None
        early_start = None
        min_metric_difference = None 
        early_dataset = None
            

    # setting up snapshot summarty writer
    summary_writer = SummaryWriter(log_dir=log_dir)

    # setting up hardware settings

    if numpy.count_nonzero([int(args.use_cuda), int(args.use_mps)]) > 1:

        raise RuntimeError("""You cannot use multiple backends for training at the same time.
        Turn on option to true and others to False.
        Example: ENABLE_CPU=False; ENABLE_CUDA=True; ENABLE_MPS=False""")

    if args.use_cuda:

        device_name = "cuda" if args.use_cuda else "cpu"

        if len(args.gpu_id.strip()) != 0 and (args.gpu_id != "-"):

            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            if isinstance(args.gpu_id, str) and args.gpu_id[0].isdigit():
                device_name = "cuda:%s" % (args.gpu_id[0])
        else:
            err_logger.warning("""you didnt provide any GPU ids.
            Cuda is going to leverage all available GPUs during training.""")

    elif args.use_mps:
        device_name = "mps"

    else:
        device_name = "cpu"

    train_device = torch.device(device_name)
    num_cpu_workers = int(args.num_workers)

    if num_cpu_workers < 3:
        err_logger.warning("""You've set number of cpu threads to be less,
        than the baseline (recommended is 3).""")

    elif num_cpu_workers > 5:
        err_logger.warning("""You've set number of cpu threads to be greater,
         than it is recommended. (5 threads is usually considered as enough'""")

    # training-specific settings

    runtime_logger.debug('LOADING NETWORK... \n')

    network = utils.get_efficientnet_network(exp_config['network'])

    optimizer = utils.get_optimizer(exp_config['optimizer'], model=network)

    lr_scheduler = utils.get_lr_scheduler(
        exp_config['scheduler'], optimizer) if 'scheduler' in exp_config else None

    loss_function = utils.get_loss_from_config(loss_config=exp_config['loss'])

    evaluation_metric = utils.get_evaluation_metric_by_name(exp_config['eval_metric'])
    
    batch_size = int(exp_config['batch_size'])
    max_epochs = int(exp_config['max_epochs'])

    # setting up classifier training pipeline

    trainer = models.NetworkTrainer(
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
        loader_num_workers=num_cpu_workers,
        early_start=early_start,
        early_stop_dataset=early_dataset,
        minimum_metric_difference=min_metric_difference,
        early_patience=early_patience
    )

    # training classifier
    runtime_logger.debug("TRAINING NETWORK ON TRAINING DATASET... \n")
    training_loss, loss_history = trainer.train(train_dataset=train_dataset)

    # plot of the loss function change during training

    loss_figure = plt.figure(figsize=(30, 30))
    _, ax = plt.subplots()
    ax.plot(numpy.arange(len(loss_history)), loss_history)

    ax.set(xlabel='Epoch', ylabel='Loss',
        title='Loss change during training')
    ax.grid()

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
    runtime_logger.debug("EVALUATING NETWORK ON VALIDATION DATASET... \n")

    evaluation_metric = trainer.evaluate(
        validation_dataset=validation_dataset
    )
    # saving evaluation metric results to summary writer
    summary_writer.add_scalar(
        "evaluation metric: ",
        scalar_value=evaluation_metric,
        double_precision=True
    )
     
    # exporting model to .ONNX format and saving to 'output_dir' path
    
    model_path = os.path.join(
        output_dir, 
        "model.onnx"
    )

    test_img, _ = validation_dataset[0]
    try:
        torch.onnx.export(
            model=trainer.network.cpu(), 
            args=test_img.unsqueeze(0).cpu(),
            f=model_path
        )
    except(UserWarning) as err:
        err_logger.warn(err)

    print('training completed.')


if __name__ == '__main__':
    training_pipeline()

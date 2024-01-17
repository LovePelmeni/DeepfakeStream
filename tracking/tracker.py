import torch 
import pandas
import logging 
import os
import typing
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentTracker(object):
    """
    Class, responsible for tracking and 
    storing information about conducted experiments.
    
    Parameters:
    ------------
    """

    experiment_root_path: str
    model_root_path: str
    data_root_path: str

    def __init__(self, 
        model_format: typing.Literal['onnx', 'pt', 'pth'],
        data_format: typing.Literal['jpeg', 'png', 'we']
    ):
        self.model_format = model_format 
        self.data_format = data_format

    def save_experiment(self, 
        network_path: str, 
        dataset_path: str,
        loss: float,
        metric: float,
        total_epochs: int,
        train_device: typing.Union[str, torch.device],
        batch_size: int,
        optimizer: str,
        lr_scheduler: typing.Optional[str]
    ):
        exp_path = os.path.join(self.experiment_root_path, "experiment_%s.pth")
        exp_model_path = os.path.join(self.model_root_path, network_path)
        exp_data_path = os.path.join(self.data_root_path, dataset_path)
        exp_date = datetime.now().strftime("%d/%m/%Y %H:%M"),
        try:
            torch.save(
                obj={
                    'data_path': os.path.join(self.data_root_path, exp_data_path),
                    'model_path': os.path.join(self.model_root_path, exp_model_path),
                    'total_epochs': total_epochs,
                    'batch_size': batch_size,
                    'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler if lr_scheduler else "-",
                    "train_device": train_device,
                    "gpus_utilized": 0 if train_device != 'cuda' else torch.cuda.device_count(),
                }, f=exp_path
            )
        except(Exception) as err:
            logger.error(err)

        # updating CSV Experiments information
        experiments_info = pandas.read_csv("../experiments/experiments.csv")
        exp_record = pandas.Series(
            data={
                'experiment_path': exp_path,
                'device_type': train_device,
                'device_name': torch.cuda.get_device_properties(0).name if train_device == 'cuda' else "CPU",
                'loss': loss,
                'metric': metric,
                'date': exp_date,
            }
        )
        # adding record with information about new experiment
        experiments_info = pandas.concat([experiments_info, exp_record], axis=0)

        # saving CSV file
        experiments_info.to_csv(path_or_buf="../experiments/experiments.csv")
        




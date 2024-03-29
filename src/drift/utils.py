import numpy 
import typing
import logging
import pathlib
import os
from datetime import datetime

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename='dataset_baseline_logs.log')
logger.addHandler(logger)

class BaselineEvaluationDataset(object):
    """
    Storage module that maps evaluation dataset
    from file storage to numpy.ndarray.

    Parameters:
    -----------
        baseline_dataset - numpy.ndarray of [label, data_obj] format.
        destination_path - folder to store evaluation dataset to.
    """
    def save(self, 
        destination_path: typing.Union[str, pathlib.Path],
        baseline_dataset: numpy.ndarray
    ):
        if not os.path.exists(destination_path):
            os.makedirs(destination_path, exist_ok=True)
        try:
            current_date = datetime.now().strftime("%d/%m/%Y")
            experiment_path = os.path.join(
                destination_path, "baseline_%s.dat" % current_date)
            baseline_dataset.tofile(fid=experiment_path)
        except(FileNotFoundError, Exception) as err:
            logger.error(err)
            raise RuntimeError("Failed to save baseline data")

    @classmethod
    def from_config(cls, config: typing.Dict[str, typing.Any]):
        try:
            dataset_path = config.get("dataset_path")
            dtype = config.get("dtype", numpy.uint8)
            reshape_form = config.get("shape", None)
            cls.dataset = numpy.memmap(
                filename=dataset_path, 
                dtype=dtype, 
                shape=reshape_form
            )
            return cls()
        except(Exception) as err:
            logger.error(err)
            raise RuntimeError("failed load data from config settings.")
        
    def map(self, obj_idx: int):
        """
        Maps object from dataset by it's index.
        """
        if obj_idx > self.dataset.shape[0]:
            raise IndexError("index '%s' is out of range. Total Dataset length is '%s'" % (
            obj_idx, self.dataset.shape[0]))

        label = self.dataset[obj_idx, 0]
        data_obj = self.dataset[obj_idx, 1]
        return label, data_obj

    @property
    def dataset(self):
        return self.dataset
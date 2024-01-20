import unittest 
import os
from src.pipelines.preproc_pipeline import data_pipeline

class DataPipelineTestCase(unittest.TestCase):

    def setUp(self):
        os.environ.setdefault("DATA_DIR", "experiments/experiment1/data/train_data")
        os.environ.setdefault("DATA_CONFIG_DIR", "data_configs/exp1_data_config.json")
        os.environ.setdefault("AUGMENTED_DATA_DIR", "experiments/experiment1/data/augmented/train_data")

    def test_data_pipeline(self):
        data_pipeline()
        self.assertTrue(os.path.exists(path="experiments/experiment1/data/augmented/train_data"))

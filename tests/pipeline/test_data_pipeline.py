from src.pipelines.preproc_pipeline import data_pipeline
import unittest 
import os
import pathlib

class DataPipelineTestCase(unittest.TestCase):

    def setUp(self):
        self.crop_dir = pathlib.Path("")
        self.config_dir = pathlib.Path("")
        self.video_data_dir = pathlib.Path("")
        os.environ.setdefault("DATA_DIR", self.video_data_dir)
        os.environ.setdefault("DATA_CONFIG_DIR", self.config_dir)
        os.environ.setdefault("AUGMENTED_DATA_DIR", self.video_data_dir)

    def test_data_pipeline(self):
        data_pipeline()
        self.assertTrue(
            os.path.exists(
                path="experiments/experiment1/data/augmented/train_data"
            )
        )



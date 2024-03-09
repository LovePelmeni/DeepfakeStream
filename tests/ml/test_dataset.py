from src.training.datasets import datasets 
from src.preprocessing.face_detector import VideoFaceDataset
from src.exceptions import exceptions

import os
import unittest
from torch.utils import data
import torch
import numpy
import pytest

@pytest.fixture(scope='module')
def invalid_video_data():
    total_samples = 10
    random_names = ["img%s" % idx for idx in range(total_samples)]
    return [
        os.path.join("test_images", random_name) 
        for random_name in random_names
    ], numpy.random.choice(size=total_samples, a=[0, 1]).tolist()

@pytest.fixture(scope='module')
def invalid_image_data():
    total_samples = 10
    random_names = ["img%s" % idx for idx in range(total_samples)]
    return [
        os.path.join("test_images", random_name) 
        for random_name in random_names
    ], numpy.random.choice(size=total_samples, a=[0, 1]).tolist()

class DeepFakeDatasetTestCase(unittest.TestCase):

    def setUp(self):
        self.test_dataset = datasets.DeepfakeDataset()
        self.batch_size = 32
        self.loader = data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def test_assert_invalid_image_format(self, invalid_image_data):
        invalid_image_paths, invalid_image_labels = invalid_image_data
        with self.assertRaises(
            expected_exception=exceptions.InvalidSourceError, 
            msg="""Dataset failed to spot invalid image paths, passed as a data source.
            Exception should be raised indicating, that image data source is invalid"""
        ):
            _ = datasets.DeepfakeDataset(
                image_paths=invalid_image_paths,
                image_labels=invalid_image_labels,
            )

    def test_dataset_return(self):
        for data in self.loader:
            self.assertEqual(len(data), self.expected_dataset_output_len)
            self.assertIn(
                member=type(torch.Tensor), 
                container=[type(instance) for instance in data],
                msg='dataset output does not contain required instances of type torch.Tensor')

class VideoDatasetTestCase(unittest.TestCase):

    def setUp(self):
        video_paths = []
        video_labels = []
        
        self.video_dataset = VideoFaceDataset(
            video_paths=video_paths,
            video_labels=video_labels,
            frames_per_vid=0.2
        )

    def test_assert_invalid_video_format(self, invalid_video_data):
        invalid_dataset = VideoFaceDataset(
                video_paths=invalid_video_data[0],
                video_labels=invalid_video_data[1],
                frames_per_vid=numpy.random.randint(
                    low=0.1, 
                    high=1, 
                    size=1
                )
        )
        output_frames = invalid_dataset.extract_frames(video_path=invalid_video_data[0][0])
        self.assertIsInstance(output_frames, type(list()))
        self.assertEqual(len(output_frames), 0)
        
    def test_dataset_output(self):
        number_of_batches = 0

        for frames in self.loader:
            self.assertIsNot(len(frames) == 0,
            msg='dataset returned empty list of frames, while it should at least output 1')
            self.assertEqual(len(frames), self.frames_per_vid)
            self.assertIsInstance(frames[0], numpy.ndarray)
            number_of_batches += 1

        self.assertEqual(
            first=number_of_batches,
            second=(len(self.video_paths)//self.batch_size) + (len(self.video_paths)%self.batch_size),
            msg='mismatch between expected number of batches to return and what was returned by the loader'
        )
        
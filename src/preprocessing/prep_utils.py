import os
from glob import glob
import json


def get_video_paths(source_path: str):
    paths = []
    for path in os.listdir(source_path):
        video_path = os.path.join(source_path, path)
        paths.append(video_path)
    return paths


def get_originals_without_fakes(root_dir: str):
    """
    Return list of original videos, which 
    do not have any deepfaked analogies.

    NOTE:
        do not misinterpret them with original videos
        which does have their respective deepfaked videos.

    Parameters:
    ----------- 

    root_dir - directory, containing files

    """
    paths = []
    for metadata_url in glob(root_dir=root_dir, pathname="**/*.json", recursive=True):

        metadata_path = os.path.join(root_dir, metadata_url)
        with open(metadata_path, mode='r') as json_file:

            config = json.load(json_file)
            base_path = '/'.join(metadata_path.split('/')[:-1])

            for video_id, row_config in config.items():
                original = row_config.get("original", None)
                video_path = os.path.join(base_path, video_id)
                if not original:
                    if os.path.exists(video_path):
                        paths.append(video_path)

        json_file.close()
    return paths


def get_fakes_without_originals(root_dir: str):
    """
    Returns deep faked videos without returning
    their respective original versions

    Parameters:
    ----------
    root_dir - directory, containing deep faked videos
    """
    paths = []
    for metadata_url in glob(root_dir=root_dir, pathname="**/*.json", recursive=True):

        metadata_path = os.path.join(root_dir, metadata_url)
        with open(metadata_path, mode='r') as json_config:

            config = json.load(json_config)
            for video_id, row_config in config.items():

                if 'original' in row_config:
                    video_path = os.path.join(root_dir, video_id)
                    if os.path.exists(video_path):
                        paths.append(video_path)

        json_config.close()
    return paths


def get_orig_fake_pairs(root_dir: str):

    pairs = []
    for metadata_url in glob(root_dir=root_dir, pathname="**/*.json", recursive=True):

        metadata_path = os.path.join(root_dir, metadata_url)
        base_path = '/'.join(metadata_url.split('/')[:-1])

        with open(metadata_path, mode='r') as json_file:
            config = json.load(json_file)

            for fake_id, row_config in config.items():
                original = row_config.get("original", None)

                if original is not None:
                    original_url = os.path.join(base_path, original)
                    fake_url = os.path.join(base_path, fake_id)
                    pair = (original_url, fake_url)
                    pairs.append(pair)
        json_file.close()
    return pairs

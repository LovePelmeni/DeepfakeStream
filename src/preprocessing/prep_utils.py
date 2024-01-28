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

    for metadata in glob(pathname=os.path.join(root_dir)):
        with open(metadata, mode='r') as json_file:

            config = json.load(json_file)
            base_path = '/'.join(metadata.split('/')[:-1])

            for video_id, row_config in config.items():
                original = row_config.get("original", None)
                if not original:
                    paths.append(os.path.join(base_path, video_id))
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
    for metadata in glob(pathname=os.path.join(root_dir, "")):
        with open(metadata, mode='r') as json_config:
            config = json.load(json_config)
            for video_id, row_config in config.items():
                if 'original' in row_config:
                    paths.append(video_id)
        json_config.close()

def get_orig_fake_pairs(root_dir: str):

    pairs = []
    for metadata in glob(pathname=os.path.join(root_dir, "*/metadata.json")):

        base_path = '/'.join(metadata.split('/')[:-1])

        with open(metadata, mode='r') as json_file:
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



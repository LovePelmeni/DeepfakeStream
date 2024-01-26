import subprocess 
from src.preprocessing import prep_utils
import argparse
import cv2 
import numpy
import logging
import multiprocessing
from os import cpu_count
from tqdm import tqdm
from functools import partial
import typing

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

Logger = logging.getLogger("comp_logger")
handler = logging.StreamHandler()
Logger.addHandler(handler)

def compress_videos(video_paths: typing.List[str]):
    for path in video_paths:
        try:
            compression_level = numpy.random.choice([23, 28, 32])
            command = "ffmpeg -i {} -c:v libx264 -crf {} -threads 1 {}".format(path, compression_level, )
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except(Exception) as err:
            Logger.debug("failed to process video: %s" % err)

def main():

    parser = argparse.ArgumentParser(description="Video Compression pipeline")
    parser.add_argument(
        name_or_flags='--videos-dir', 
        dest='videos_dir', 
        help='directory, containing .mp3 / .mp4 videos', 
        type=str
    )
    try:
        args = parser.parse_args()
        loaded_videos = prep_utils.get_video_paths(args.videos_dir)

    except(FileNotFoundError):
        raise SystemExit("Invalid videos source path.")
    
    except(Exception) as err:
        raise SystemExit(err)

    with multiprocessing.Pool(processes=cpu_count() - 3) as pool:
        with tqdm(total=len(loaded_videos)) as tq:
            for _ in pool.imap_unordered(
                partial(compress_videos, root_dir=args.videos_dir), 
                loaded_videos
            ):
                tq.update()
    compress_videos(args.videos_dir)
    Logger.debug('compression completed')

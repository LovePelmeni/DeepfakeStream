import argparse
import multiprocessing 
import cv2 
import os
import pathlib
from glob import glob
import functools 
from tqdm import tqdm
import typing
import collections
import json
import numpy

def get_video_labels(root_dir: str, file_ext: str):
    """
    Returns corresponding labels for all mp4 or mp3 video files,
    presented under 'root_dir'.

    NOTE:
        you have to create config file with extension of 'file_ext'
        where labels are already specified.
        This function is only responsible for extracting them 
        and presenting for further usage. Not figuring out labels
        for images on it's own.
    RETURNS:
        dictionary, containing labels for videos
    """
    output_labels =  collections.defaultdict(str)
    for metadata_url in glob(pathname=f"**/*.{file_ext}", root_dir=root_dir):

        try:
            config = json.load(metadata_url)
        except(Exception):
            print('failed to parse .json config')
            continue
        
        for video_name, info in config.items():

            if 'original' in info:
                output_labels[video_name] = "FAKE"
            else:
                output_labels[video_name] = "ORIG"

    return output_labels


def extract_frame(
    source_path: str, 
    save_path: str, 
    video_name: str, 
    video_label: typing.Literal['ORIG', 'FAKE']
):
    """
    Extracts first from the video (mp4, mp3) file,
    then saves it to the destination path
    """
    video_path = os.path.join(source_path, video_name)

    if not os.path.exists(video_path): 
        print("\n Directory '%s' does not exist \n")
        return 

    buffer = cv2.VideoCapture(video_path)

    success, curr_frame = buffer.read()
    if not success: 
        print('\n failed to read video: %s \n' % video_name)
        return

    video_name = os.path.splitext(os.path.basename(video_name))[0]
    save_path = os.path.join(save_path, video_name + "_%s.jpeg" % video_label)
    
    # saving frame, in case we haven't extracted any frames from this video before
    if not os.path.exists(save_path):
        cv2.imwrite(filename=save_path, img=curr_frame)

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument 
    arg("--video-data-dir", type=str, dest='video_data_dir', help='directory, containing video files for processing, available formats: (mp4, mp3)')
    arg("--output-img-dir", type=str, dest="output_img_dir", help='directory to store cropped video frames')
    args = parser.parse_args()

    video_data_dir = pathlib.Path(args.video_data_dir)
    output_img_dir = pathlib.Path(args.output_img_dir)
    
    # initializing directory for storing video frames
    os.makedirs(output_img_dir, exist_ok=True)
    
    # collecting video paths for the source directory
    video_paths = numpy.asarray(
    glob(
        root_dir=video_data_dir,
        pathname="**/*.mp4",
        recursive=True
    ) + glob(
        root_dir=video_data_dir,
        pathname="**/*.mp3",
        recursive=True
    )).astype(numpy.object_)

    video_labels = get_video_labels(root_dir=video_data_dir)
    video_information = collections.defaultdict(dict)

    # merging labels and their respective paths

    for video in range(len(video_paths)):
        name = os.path.splitext(os.path.basename(video_paths[video].str))[0]

        if video_labels.get(name, None) != None:
            video_information[name]['label'] = video_labels[name]
            video_information[name]['path'] = video_paths[video].str

    print('total number of videos found: %s' % len(video_paths))
    print('total number of video labels found: %s' % len(video_labels))
    
    with multiprocessing.Pool(processes=os.cpu_count()-2) as pool:
        with tqdm(desc='extracting video frames....') as tq:
            for _ in pool.imap_unordered(
                func=functools.partial(
                    extract_frame, 
                    video_data_dir, 
                    output_img_dir
                ), iterable=video_paths
            ):
                tq.update()

if __name__ == '__main__':
    main()
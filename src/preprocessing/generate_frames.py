import argparse
import multiprocessing 
import cv2 
import os
import pathlib
from glob import glob
import functools 
from tqdm import tqdm

def extract_frame(source_path: str, save_path: str, video_name: str):
    """
    Extracts first from the video (mp4, mp3) file,
    then saves it to the destination path
    """
    video_path = os.path.join(source_path, video_name)
    if not os.path.exists(video_path): raise FileNotFoundError("Directory does not exist")
    buffer = cv2.VideoCapture(video_path)

    success, curr_frame = buffer.read()
    if not success: raise ValueError('failed to read video: %s' % video_name)

    video_name = os.path.splitext(os.path.basename(video_name))[0]
    save_path = os.path.join(save_path, video_name + ".jpeg")
    
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
    video_paths = glob(
        root_dir=video_data_dir,
        pathname="**/*.mp4",
        recursive=True
    ) + glob(
        root_dir=video_data_dir,
        pathname="**/*.mp3",
        recursive=True
    )
    
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
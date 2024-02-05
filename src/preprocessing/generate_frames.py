import argparse
import multiprocessing
import os
import pathlib
from glob import glob
import functools
from tqdm import tqdm
import cv2
import numpy


def extract_frame(source_path: str, save_path: str, video_name: str):
    """
    Extracts first from the video (mp4, mp3) file,
    then saves it to the destination path
    """
    video_path = os.path.join(source_path, video_name)

    if not os.path.exists(video_path):
        print("\n path '%s' does not exist \n" % video_name)
        return

    buffer = cv2.VideoCapture(video_path)

    success, curr_frame = buffer.read()

    if not success:
        print('\n failed to read video: %s \n' % video_name)
        return

    video_name = os.path.splitext(os.path.basename(video_name))[0]
    save_path = os.path.join(save_path, video_name + ".jpeg")
    
    # saving frame, in case we haven't extracted any frames from this video before
    if not os.path.exists(save_path):
        cv2.imwrite(filename=save_path, img=curr_frame)

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument 
    arg("--video-data-dir", type=str, required=True, dest='video_data_dir', help='directory, containing video files for processing, available formats: (mp4, mp3)')
    arg("--output-img-dir", type=str, required=True, dest="output_img_dir", help='directory to store cropped video frames')
    arg("--total-frames", type=int, required=True, dest='total_frames', help='total number of frames to extract from the available dataset videos. (1 frame per video), hence N frames per N videos.')
    args = parser.parse_args()

    video_data_dir = pathlib.Path(args.video_data_dir)
    output_img_dir = pathlib.Path(args.output_img_dir)
    total_frames = int(args.total_frames)
    
    # initializing directory for storing video frames
    os.makedirs(output_img_dir, exist_ok=True)
    
    # collecting video paths for the source directory
    video_paths = numpy.asarray(
            glob(
            os.path.join(video_data_dir,
            "**/*.mp4"),
            recursive=True
        ) + glob(
            os.path.join(video_data_dir,
            "**/*.mp3"),
            recursive=True
        )
    ).astype(numpy.object_)

    if len(video_paths) == 0:
        raise SystemExit("Failed to find any videos under the provided folder '%s', check the correctness of the path" % video_data_dir)

    print("total number of videos found: %s" % len(video_paths))

    # extracting random video frames from the dataset
    random_indices = numpy.random.randint(low=0, high=len(video_paths)-1, size=total_frames)
    video_paths = video_paths[random_indices]

    print("total number of videos extracted: %s" % len(video_paths))

    with multiprocessing.Pool(processes=os.cpu_count()-2) as pool:
        with tqdm(desc='video files processed', colour='GREEN') as tq:
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
from src.preprocessing import prep_utils 
from os import cpu_count 
from torch.utils import data

import cv2 
from src.preprocessing.face_detector import VideoFaceDataset, MTCNNFaceDetector

import argparse
import os
import json
from tqdm import tqdm 
import pathlib

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def extract_faces(
    video_frame_dataset: data.Dataset, 
    image_size: int, 
    output_dir: str
):
    face_detector = MTCNNFaceDetector(
        image_size=image_size,
        use_landmarks=False,
        keep_all_pred_faces=False,
        min_face_size=160, # change this parameter, in case minimum face size is different
    )

    loader = data.DataLoader(
        batch_size=1,
        dataset=video_frame_dataset,
        shuffle=True,
        num_workers=max(max(cpu_count(), 0)-2, 0),
        collate_fn=lambda x: x
    )

    curr_video = 1
    for frames in tqdm(loader, desc='video: %s' % str(curr_video)): # per video

        # iterating over frames of 1 video and extracting crops
        for _, frame_img in frames.items():

            output_path = os.path.join(output_dir, "video_%s_faces.json" % str(curr_video))
            faces = face_detector.detect_faces(frame_img)

            numerated_predictions = {idx: box for idx, box in zip(range(len(faces)), faces)}
            json.dump(numerated_predictions, fp=output_path)
            curr_video += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        name_or_flags="--data-dir", 
        dest='videos_dir', 
        type=str, help='Directory, containing video files'
    )
    parser.add_argument(
        name_or_flags='--data-config-dir',
        type=str,
        dest='data_config_dir',
        help='path to the .json config file, containing information about the image dataset'
    )
    parser.add_argument(
        name_or_flags="--crop-dir",
        dest="crop_dir",
        type=str,
        help='directory for storing faces, cropped from the image'
    )

    args = parser.parse_args()
    
    video_paths = prep_utils.get_video_paths(args.videos_dir)
    video_dataset = VideoFaceDataset(video_paths=video_paths)

    # creating directory for storing cropped faces
    os.makedirs(args.crop_dir, exist_ok=True)

    config_dir = pathlib.Path(args.data_config_dir)
    image_dataset_info = json.load(fp=config_dir)
    
    # processing images and extracting human faces crops from them.
    extract_faces(
        video_frame_dataset=video_dataset, 
        image_size=image_dataset_info.get("image_size"), 
        crop_dir=args.crop_dir
    )


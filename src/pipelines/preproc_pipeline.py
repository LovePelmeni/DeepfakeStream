import src.pipelines.train_utils as utils
from src.preprocessing import augmentations
from src.preprocessing.augmentations import resize
from src.preprocessing.face_detector import VideoFaceDataset, MTCNNFaceDetector
from src.preprocessing import prep_utils

import pandas
import argparse
import pathlib
import logging
import sys
from torch.utils import data
import os
from tqdm import tqdm
import cv2
import numpy

"""
Data preprocessing pipeline
for automating processes of cleansing,
transforming and loading data at different scales.

Interaction is available via basic CLI utility.
"""

runtime_logger = logging.getLogger("preproc_pipeline_logger")
err_logger = logging.getLogger(name="preproc_pipeline_err_logger")

runtime_logger.setLevel(level=logging.DEBUG)
err_logger.setLevel(level=logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

runtime_handler = logging.StreamHandler(stream=sys.stdout)
err_handler = logging.FileHandler(filename="preproc_pipeline_error_logs.log")

err_handler.setFormatter(formatter)
runtime_handler.setFormatter(formatter)

runtime_logger.addHandler(runtime_handler)
err_logger.addHandler(err_handler)


def data_pipeline():

    runtime_logger.debug('\n \n1. running preprocessing pipeline... \n')

    parser = argparse.ArgumentParser(
    description="CLI-based Data Preprocessing Pipeline")
    arg = parser.add_argument

    arg("--data-dir", type=str, dest='data_dir',
    required=True, help='path, containing original and fake videos.')

    arg('--json-data-config-path', type=str, dest='prep_config_path', required=True,
    help='.json file, containing extra information about preprocessing. \
        Information for face detector, frames per video to extract, etc...')

    arg('--csv-labels-crop-path', type=str, dest='labels_crop_path', 
    required=True, help='path to save labels for cropped faces')

    arg("--crop-dir", type=str, dest='crop_dir',
    required=True, help='destination path for storing cropped faces')

    arg('--dataset-type', 
    type=str, 
    choices=['train', 'validation'], 
    dest='dataset_type', required=True, 
    help='type of the dataset "train" or "validation"')

    runtime_logger.debug('2. parsing arguments... \n')
    args = parser.parse_args()

    # parsing directories and other data arguments

    config_dir = pathlib.Path(args.prep_config_path)
    data_dir = pathlib.Path(args.data_dir)
    
    output_labels_crop_dir = pathlib.Path(args.labels_crop_path)
    output_crop_dir = pathlib.Path(args.crop_dir)

    # initializing crop output directories for crops and it's labels

    os.makedirs(output_labels_crop_dir, exist_ok=True)
    os.makedirs(output_crop_dir, exist_ok=True)

    runtime_logger.debug('3. loading configuration files... \n')

    img_config = utils.load_config(config_path=config_dir)

    dataset_type = args.dataset_type
    min_face_size = img_config.get("min_face_size", 160)
    frames_per_vid_ratio = img_config.get("frames_per_vid_ratio") # percentage of videos to extract from each video

    try:
        mtcnn_img_height = img_config.get("mtcnn_image_size")  # height of the image
        mtcnn_img_width = img_config.get("mtcnn_image_size")  # width of the image
    except(KeyError):
        raise SystemExit("""you did not provide 'mtcnn_image_size' parameter,
            as it is required for fetching faces from the video frames. 
            The 'mtcnn_image_size' is basically the resolution 
            of the videos you have in your dataset.""")
    try:
        encoder_image_height = img_config.get("encoder_image_size")
        encoder_image_width = img_config.get("encoder_image_size")
    except(KeyError):
        raise SystemExit("""You didn't specified 'encoder_image_size' parameter, \
            which is stands for the size of the output cropped faces""")

    runtime_logger.debug('4. initializing augmentations \n')

    # picking augmentations

    if dataset_type.lower() == "train":

        augments = augmentations.get_training_augmentations(
            HEIGHT=mtcnn_img_height,
            WIDTH=mtcnn_img_width,
        )

    elif dataset_type.lower() == "validation":

        augments = augmentations.get_validation_augmentations(
            HEIGHT=mtcnn_img_height,
            WIDTH=mtcnn_img_width
        )

    runtime_logger.debug('5. applying transformations... \n \n')


    # extracting video paths from the original and fake video paths

    original_videos = numpy.asarray(prep_utils.get_originals_without_fakes(root_dir=data_dir))
    fake_videos = numpy.asarray(prep_utils.get_fakes_without_originals(root_dir=data_dir))

    output_video_paths = numpy.concatenate([original_videos, fake_videos])
    output_video_labels = numpy.concatenate(
        [
            numpy.repeat([0], len(original_videos)), 
            numpy.repeat([1], len(fake_videos))
        ]
    )
    
    if (len(output_video_paths.flatten()) == 0) or (len(output_video_labels.flatten()) == 0):
        raise SystemExit("Failed to find unique images for processing.")

    video_dataset = VideoFaceDataset(
        video_paths=output_video_paths, 
        video_labels=output_video_labels,
        frames_per_vid=frames_per_vid_ratio # 1 percent of frames from the video is selected to avoid duplicates
    )

    video_loader = data.DataLoader(
        dataset=video_dataset,
        shuffle=False,
        num_workers=max(os.cpu_count()-2, 0),
        batch_size=1, # one video per iteration
    )

    face_detector = MTCNNFaceDetector(
        image_size=mtcnn_img_height, 
        use_landmarks=True, 
        keep_all_pred_faces=False,
        min_face_size=min_face_size,
        inf_device="cpu",
    )

    # processing label information

    output_labels = pandas.DataFrame(columns=["crop_path", "label", "video_id"])

    # processing videos 

    curr_video = 0

    for label, ext_frames in tqdm(video_loader, desc="video #%s: " % str(curr_video)):

        video_id = os.path.splitext(
            os.path.basename(
                output_video_paths[curr_video]
                )
        )[0]

        for frame_idx in range(len(ext_frames)):

            # augmenting video frame

            video_frame = ext_frames[frame_idx]
            print(video_frame.numpy().shape)
            
            augmented_frame = augments(image=video_frame.numpy())['image']

            # predicting faces bounding boxes and landmarks 
            face_boxes, _ = face_detector.detect_faces(augmented_frame)
            
            # matching bounding boxes between each other 
          
            for box_idx in range(len(face_boxes)):
                
                print(face_boxes[box_idx])
                
                ox1, oy1, ox2, oy2 = face_boxes[box_idx]

                cropped_face = augmented_frame[ox1:ox2, oy1:oy2]

                # resizing face crop, according to the encoder input requirements

                resize_face_crop = resize.IsotropicResize(
                    target_shape=(encoder_image_height, encoder_image_width)
                )
 
                resized_face = resize_face_crop(image=cropped_face)['image']

                face_file_name = "{}_{}_{}.png".format(str(video_id), str(frame_idx), str(box_idx))
                frame_path = os.path.join(output_crop_dir, face_file_name)

                # saving extracted fake and original faces
                cv2.imwrite(filename=frame_path, img=resized_face)

                row = pandas.Series(
                    data={
                        'crop_path': frame_path, 
                        'label': label, 
                        'video_id': video_id
                    }
                )
                output_labels = pandas.concat([output_labels, row])

        curr_video += 1

    if len(output_labels) == 0: 
        raise SystemExit(
            """none of the videos, we managed to find, 
            according to presented CSV label config, does exist.""")
            
    # saving final crop output labels
    output_labels.to_csv(path_or_buf=output_labels_crop_dir)

    print("processing pipeline completed")

if __name__ == '__main__':
    data_pipeline()


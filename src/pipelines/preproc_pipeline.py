import src.pipelines.train_utils as utils
from src.preprocessing import augmentations
from src.preprocessing.augmentations import resize
from src.preprocessing.face_detector import VideoFaceDataset, MTCNNFaceDetector
from src.preprocessing.prep_utils import get_video_paths

import argparse
import pathlib
import logging
import sys
from torch.utils import data
import os
from tqdm import tqdm
import cv2

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

    arg("--orig-data-dir", type=str, dest='orig_data_dir',
    required=True, help='path to original videos')

    arg('--fake-data-dir', 
    type=str, dest='deepfake_data_dir', 
    required=True, help='path to deepfaked videos')

    arg('--data-config-dir', type=str, dest='config_dir', required=True,
    help='configuration .json file, containing information about the data')

    arg("--orig-crop-dir", type=str, dest='orig_crop_dir',
    required=True, help='path to save orig cropped faces')

    arg("--fake-crop-dir", type=str, dest='fake_crop_dir',
    required=True, help='path to save fake cropped faces')

    arg('--dataset-type', type=str, choices=['train', 'validation'], 
    dest='dataset_type', required=True, 
    help='type of the dataset "train" or "validation"')

    runtime_logger.debug('2. parsing arguments... \n')
    args = parser.parse_args()

    # parsing directories and other data arguments

    config_dir = pathlib.Path(args.config_dir)

    orig_data_dir = pathlib.Path(args.orig_data_dir)
    fake_data_dir = pathlib.Path(args.deepfake_data_dir)

    orig_output_dir = pathlib.Path(args.orig_crop_dir)
    fake_output_dir = pathlib.Path(args.fake_crop_dir)

    # initializing crop output directories 

    os.makedirs(orig_output_dir, exist_ok=True)
    os.makedirs(fake_output_dir, exist_ok=True)

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

    orig_video_paths = get_video_paths(orig_data_dir)
    fake_video_paths = get_video_paths(fake_data_dir)

    video_dataset = VideoFaceDataset(
        orig_video_paths=orig_video_paths, 
        fake_video_paths=fake_video_paths,
        frames_per_vid=frames_per_vid_ratio # 1 percent of frames from the video is selected to avoid duplicates
    )

    video_loader = data.DataLoader(
        orig_dataset=video_dataset,
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

    # processing videos 

    curr_video = 0

    for orig_frames, fake_frames in tqdm(video_loader, desc="video #%s: " % str(curr_video)):

        video_id = os.path.splitext(os.path.basename(orig_video_paths[curr_video]))[0]

        orig_video_dir = os.path.join(orig_output_dir, video_id)
        fake_video_dir = os.path.join(fake_output_dir, video_id)

        for frame_idx in range(len(orig_frames)):

            # augmenting original 
            orig_frame = orig_frames[frame_idx]
            fake_frame = fake_frames[frame_idx]

            augmented_frame = augments(image=orig_frame)['image']

            # predicting faces bounding boxes and landmarks 
            orig_face_boxes, _ = face_detector.detect_faces(augmented_frame)
            fake_face_boxes, _ = face_detector.detect_faces(fake_frame)
            
            # matching bounding boxes between each other 
          
            for box_idx in range(len(orig_face_boxes)):

                ox1, oy1, ox2, oy2 = orig_face_boxes[box_idx]
                fx1, fy1, fx2, fy2 = fake_face_boxes[box_idx]

                orig_cropped_face = augmented_frame[ox1:ox2, oy1:oy2]
                fake_cropped_face = fake_frame[fx1:fx2, fy1:fy2]

                # resizing face crop, according to the encoder input requirements

                resize_face_crop = resize.IsotropicResize(
                    target_shape=(encoder_image_height, encoder_image_width)
                )
 
                resized_orig_face = resize_face_crop(image=orig_cropped_face)['image']
                resized_fake_face = resize_face_crop(image=fake_cropped_face)['image']

                orig_frame_path = os.path.join(orig_video_dir, "{}_{}.png".format(str(frame_idx), str(box_idx)))
                fake_frame_path = os.path.join(fake_video_dir, "{}_{}.png".format(str(frame_idx), str(box_idx)))

                # saving extracted fake and original faces
                cv2.imwrite(filename=orig_frame_path, img=resized_orig_face)
                cv2.imwrite(filename=fake_frame_path, img=resized_fake_face)

        curr_video += 1

    print("processing pipeline completed")

if __name__ == '__main__':
    data_pipeline()



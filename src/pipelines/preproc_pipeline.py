import src.pipelines.train_utils as utils
from src.preprocessing import augmentations
from src.preprocessing.augmentations import resize
from src.preprocessing.face_detector import VideoFaceDataset, MTCNNFaceDetector
from src.preprocessing import prep_utils

import cv2
import pandas
import argparse
import pathlib
import logging
import sys
from torch.utils import data
import os
from tqdm import tqdm
import numpy
import torch

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

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

runtime_handler = logging.StreamHandler(stream=sys.stdout)
err_handler = logging.FileHandler(filename="preproc_pipeline_error_logs.log")

err_handler.setFormatter(formatter)
runtime_handler.setFormatter(formatter)

runtime_logger.addHandler(runtime_handler)
err_logger.addHandler(err_handler)


def data_pipeline():

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

    args = parser.parse_args()

    # parsing directories and other data arguments

    config_dir = pathlib.Path(args.prep_config_path)
    data_dir = pathlib.Path(args.data_dir)

    output_labels_crop_dir = pathlib.Path(args.labels_crop_path)
    output_crop_dir = pathlib.Path(args.crop_dir)

    # initializing crop output directories for crops and it's labels

    os.makedirs(output_labels_crop_dir.parent, exist_ok=True)
    os.makedirs(output_crop_dir, exist_ok=True)

    runtime_logger.debug('3. loaded configuration files! \n')

    img_config = utils.load_config(config_path=config_dir)

    dataset_type = args.dataset_type

    # device to use during inference of the network (default is "CPU")
    inference_device = img_config.get("inference_device", "cpu")

    # minimum possible size of the human face to detect on the images
    min_face_size = img_config.get("min_face_size", 160)

    # percentage of videos to extract from each video
    frames_per_vid_ratio = img_config.get("frames_per_vid_ratio")

    try:
        mtcnn_input_size = img_config.get(
            "mtcnn_image_size")  # height of the image
    except (KeyError):
        raise SystemExit("""you did not provide 'mtcnn_image_size' parameter,
            as it is required for fetching faces from the video frames. 
            The 'mtcnn_image_size' is basically the resolution 
            of the videos you have in your dataset.""")
    try:
        encoder_image_size = img_config.get("encoder_image_size")
    except (KeyError):
        raise SystemExit("""You didn't specified 'encoder_image_size' parameter, \
            which is stands for the size of the output cropped faces""")

    # picking augmentations

    if dataset_type.lower() == "train":

        augments = augmentations.get_training_augmentations(
            IMAGE_SIZE=mtcnn_input_size)

    elif dataset_type.lower() == "validation":

        augments = augmentations.get_validation_augmentations(
            IMAGE_SIZE=mtcnn_input_size)

    # extracting video paths from the original and fake video paths

    original_videos = numpy.asarray(
        prep_utils.get_originals_without_fakes(root_dir=data_dir))
    fake_videos = numpy.asarray(
        prep_utils.get_fakes_without_originals(root_dir=data_dir))

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
        # 1 percent of frames from the video is selected to avoid duplicates
        frames_per_vid=frames_per_vid_ratio
    )

    video_loader = data.DataLoader(
        dataset=video_dataset,
        shuffle=False,
        num_workers=max(os.cpu_count()-2, 0), # specifying the number of processing for data loading
        batch_size=1,  # one video per iteration
    )

    face_detector = MTCNNFaceDetector(
        image_size=mtcnn_input_size,
        use_landmarks=True,
        keep_all_pred_faces=False, # do not keep all candidates of human faces, perform NMS filtering instead
        min_face_size=min_face_size, # minimum size of human face
        inf_device=inference_device, # device to use during inference
    )

    # processing label information

    output_labels = pandas.DataFrame(
        columns=["crop_path", "label", "video_id"])

    # processing videos

    curr_video = 0

    with torch.no_grad():

        for label, ext_frames in tqdm(video_loader, desc="processing video #%s: " % str(curr_video)):

            video_id = os.path.splitext(
                os.path.basename(
                    output_video_paths[curr_video]
                )
            )[0]

            for frame_idx in range(len(ext_frames)):

                # extracting video frame from the video
                video_frame = ext_frames[frame_idx].squeeze(0).numpy()

                # converting to RGB format
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

                # augmenting video frame using set of specified augmentations
                augmented_frame = augments(image=video_frame)['image']

                # predicting faces bounding boxes and landmarks
                face_boxes, _ = face_detector.detect_faces(augmented_frame)

                # matching bounding boxes between each other

                for box_idx in range(len(face_boxes)):

                    ox1, oy1, ox2, oy2 = face_boxes[box_idx]

                    ox1 = max(round(ox1), 0)
                    ox2 = max(
                        round(ox2), 0) if ox2 < augmented_frame.shape[0] else augmented_frame.shape[0]
                    oy1 = max(round(oy1), 0)
                    oy2 = max(
                        round(oy2), 0) if oy2 < augmented_frame.shape[1] else augmented_frame.shape[1]

                    if ((ox2 - ox1) < min_face_size) or ((oy2 - oy1) < min_face_size):
                        continue

                    cropped_face = augmented_frame[
                        round(ox1):round(ox2),
                        round(oy1):round(oy2)
                    ]

                    # resizing face crop, according to the encoder input requirements

                    resize_face_crop = resize.IsotropicResize(
                        target_size=encoder_image_size,
                    )

                    resized_face = resize_face_crop(
                        image=cropped_face)['image']

                    face_file_name = "{}_{}_{}.png".format(
                        str(video_id), str(frame_idx), str(box_idx))
                    frame_path = os.path.join(output_crop_dir, face_file_name)

                    # saving extracted fake and original faces
                    cv2.imwrite(filename=frame_path, img=resized_face)

                    row = pandas.Series(
                        data={
                            'crop_path': frame_path,
                            'label': label.item(),
                            'video_id': video_id
                        }
                    )
                    output_labels = pandas.concat([output_labels, row])

            curr_video += 1

    if len(output_labels) == 0:
        raise SystemExit(
            """none of the videos, mentioned in CSV label config does exist.""")

    # saving final crop output labels
    output_labels.to_csv(path_or_buf=output_labels_crop_dir)

    print("processing pipeline completed")


if __name__ == '__main__':
    data_pipeline()

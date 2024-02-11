import src.pipelines.utils as utils
from src.preprocessing import crop_augmentations
from src.preprocessing.face_detector import VideoFaceDataset, MTCNNFaceDetector

import cv2
import pandas
import argparse
import pathlib
import logging
import sys
from torch.utils import data
import os
from tqdm import tqdm
import torch
import gc

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
        required=True, help='folder, containing original and fake videos.')

    arg("--labels-path", type=str, dest='labels_path',
        required=True, help='path to video labels csv file')

    arg('--preproc-config-path', type=str, dest='prep_config_path', required=True,
        help='.json file, containing extra information about preprocessing. \
        Information for face detector, frames per video to extract, etc...')

    arg("--output-crop-dir", type=str, dest='crop_dir',
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
    labels_path = pathlib.Path(args.labels_path)

    output_crop_dir = pathlib.Path(args.crop_dir)
    output_labels_crop_dir = pathlib.Path(os.path.join(output_crop_dir, "crop_labels.csv"))

    # initializing crop output directories for crops and it's labels

    os.makedirs(output_labels_crop_dir.parent, exist_ok=True)
    os.makedirs(output_crop_dir, exist_ok=True)

    img_config = utils.load_config(config_path=config_dir)

    dataset_type = args.dataset_type

    # device to use during inference of the network (default is "CPU")
    inference_device = img_config.get("inference_device", "cpu")

    # minimum possible size of the human face to detect on the images
    min_face_size = img_config.get("min_face_size", 160)

    # percentage of videos to extract from each video
    frames_per_vid = img_config.get("frames_per_vid")

    try:
        mtcnn_crop_margin = img_config.get(
            "mtcnn_crop_margin")  # margin to add to the cropped face
    except KeyError:
        raise SystemExit("""you did not provide 'mtcnn_image_size' parameter,
            as it is required for fetching faces from the video frames. 
            The 'mtcnn_image_size' is basically the resolution 
            of the videos you have in your dataset.""")
    try:
        encoder_image_size = img_config.get("encoder_image_size")
    except KeyError:
        raise SystemExit("""You didn't specified 'encoder_image_size' parameter, \
            which is stands for the size of the output cropped faces""")

    # picking augmentations

    if dataset_type.lower() == "train":

        augments = crop_augmentations.get_train_crop_augmentations(
            CROP_IMAGE_SIZE=encoder_image_size)

    elif dataset_type.lower() == "validation":

        augments = crop_augmentations.get_validation_crop_augmentations(
            CROP_IMAGE_SIZE=encoder_image_size)
    else:
        raise SystemExit("invalid dataset type, acceptable only 'train' or 'validation'.")

    # extracting video paths from the original and fake video paths

    labels = pandas.read_csv(labels_path)

    video_names = labels['name'].tolist()

    video_paths = [
        os.path.join(data_dir, video)
        for video in video_names
    ]

    video_labels = labels['label'].tolist()

    if len(video_paths) == 0:
        raise SystemExit("failed to find any videos from the folder you've provided. Make sure labels file is correct and folder does exist")

    video_dataset = VideoFaceDataset(
        video_paths=video_paths,
        video_labels=video_labels,
        # 1 percent of frames from the video is selected to avoid duplicates
        frames_per_vid=frames_per_vid
    )

    video_loader = data.DataLoader(
        dataset=video_dataset,
        shuffle=False,
        num_workers=max(os.cpu_count()-2, 0), # specifying the number of processing for data loading
        batch_size=1,  # one video per iteration
    )

    detector = MTCNNFaceDetector(
        margin=mtcnn_crop_margin, # use for adding margin
        use_landmarks=False,
        keep_all_pred_faces=False, # do not keep all candidates of human faces, perform NMS filtering instead
        min_face_size=min_face_size, # minimum size of human face
        inf_device=inference_device, # device to use during inference
    )

    # processing label information

    output_labels = pandas.DataFrame(
        columns=["name", "split", "label"])

    # processing videos

    curr_video = 0

    with torch.no_grad():

        with tqdm(desc='videos processed') as tq:

            for frame_labels, ext_frames in video_loader:

                video_id = os.path.splitext(video_names[curr_video])[0]

                for frame_idx in range(len(ext_frames)):

                    # extracting video frame from the video
                    video_frame = ext_frames[frame_idx].squeeze(0).numpy()

                    if (video_frame.shape[2] == 3):
                        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

                    # predicting faces bounding boxes and landmarks
                    face_boxes, _ = detector.detect_faces(video_frame)

                    # matching bounding boxes between each other

                    if len(face_boxes) != 0:
                        print("found %s faces at video: '%s', at '%s' frame." % (
                            len(face_boxes),
                            video_dataset.video_paths[curr_video],
                            frame_idx
                            )
                        )

                    for box_idx in range(len(face_boxes)):

                        ox1, oy1, ox2, oy2 = face_boxes[box_idx]

                        ox1 = max(round(ox1), 0)
                        ox2 = max(round(ox2), 0) if ox2 < video_frame.shape[0] else video_frame.shape[0]
                        oy1 = max(round(oy1), 0)
                        oy2 = max(round(oy2), 0) if oy2 < video_frame.shape[1] else video_frame.shape[1]


                        if ((ox2 - ox1) < min_face_size) or ((oy2 - oy1) < min_face_size):
                            continue

                        cropped_face = video_frame[
                            oy1:oy2,
                            ox1:ox2
                        ]

                        augmented_face = augments(image=cropped_face)['image']

                        # resizing face crop, according to the encoder input requirements

                        face_file_name = "{}_{}_{}.png".format(
                            str(video_id), str(frame_idx), str(box_idx))
                        frame_path = os.path.join(output_crop_dir, face_file_name)

                        # saving extracted fake and original faces
                        success = cv2.imwrite(filename=frame_path, img=augmented_face)
                        if success:
                            row = pandas.DataFrame(
                                data={
                                    'name': [face_file_name],
                                    'split': [dataset_type.lower()],
                                    'label': [frame_labels[0]]
                                }
                            )
                            output_labels = pandas.concat([output_labels, row])
                        else:
                            print("failed to save image file: '%s'" % face_file_name)

                curr_video += 1
                tq.update()

                del ext_frames
                gc.collect()


    print('managed to found %s faces out of %s' % (curr_video, len(video_dataset.video_paths)))

    if len(output_labels) == 0:
        raise SystemExit(
            """none of the videos, mentioned in CSV label config does exist.""")

    # saving final crop output labels
    output_labels.to_csv(path_or_buf=output_labels_crop_dir, index=False)

    # removing variables and turning on garbage collector

    del video_dataset
    del video_paths
    del video_loader
    del video_labels
    del detector
    gc.collect()
    print("processing pipeline completed")


if __name__ == '__main__':
    data_pipeline()

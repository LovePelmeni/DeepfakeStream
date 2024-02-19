import numpy
import pandas
import argparse
import pathlib
import os
import shutil


def move_videos(video_paths: list, output_path: str):
    for video_path in video_paths:
        dest_path = os.path.join(output_path, video_paths)
        shutil.move(src=video_path, dest=dest_path)


def main():

    parser = argparse.ArgumentParser()
    args = parser.add_argument

    args("--videos-path", dest='videos_path',
         type=str, help='path to videos folder')
    args("--labels-path", dest='labels_path', type=str,
         help='path to video folder labels (csv file)')
    args("--output-path", dest='output_path',
         type=str, help='folder to save videos')
    args('--output-labels-name', dest='output_labels_name', type=str,
         help='name of the CSV file to save labels (without extension)')
    args("--num-videos", dest='num_videos', type=int,
         help='number of videos to extract from the folder')
    args("--fake-prop", type=int, default=0.5, dest='fake_prop',
         help='proportion of fake videos to select')
    args("--orig-prop", type=int, default=0.5, dest='orig_prop',
         help='proportion of orig videos to select')

    params = parser.parse_args()

    videos_path = pathlib.Path(params.videos_path)
    labels_path = pathlib.Path(params.labels_path)

    output_folder = pathlib.Path(params.output_path)
    output_labels_path = pathlib.Path(
        os.path.join(params.output_folder, params.output_labels_name)
    )

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    fake_prop = int(params.fake_prop)
    orig_prop = int(params.orig_prop)
    num_videos = int(params.num_videos)

    ext = os.path.splitext(os.path.basename())[-1]

    if ext.lower() == 'json':
        labels = pandas.read_json(labels_path)

    if ext.lower() == 'csv':
        labels = pandas.read_csv(labels_path)

    fake_videos = labels[(labels['original'].isna()) &
                         (labels['label'] == 'FAKE')]
    orig_videos = labels[(labels['label'] == 'ORIG')]

    random_fake_indices = numpy.random.choice(
        a=fake_videos.index.to_list(), size=num_videos*fake_prop
    )

    random_orig_indices = numpy.random.choice(
        a=orig_videos.index.to_list(), size=num_videos*orig_prop
    )

    labels = labels.iloc[
        numpy.unique(
            random_fake_indices + random_orig_indices)
    ]

    # moving videos to new folder
    selected_videos = (
        fake_videos['name'].tolist() + orig_videos['name'].tolist())

    video_paths = [
        os.path.join(videos_path, video_name)
        for video_name in selected_videos
    ]

    # saving videos
    move_videos(video_paths, output_path=output_folder)

    # updating labels
    output_labels = pandas.read_csv(output_labels_path)
    output_labels = pandas.concat([output_labels, labels], axis=0)
    output_labels.index = numpy.arange(len(output_labels))
    output_labels.to_csv(output_labels_path)

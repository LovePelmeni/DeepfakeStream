import pandas
import argparse
import pathlib 
import os
import shutil 

def merge_labels(csv_label_output_path: str, json_curr_label_path: str):
    """
    Updates main output labels file
    with labels, loaded from another file 
    Parameters:
    -----------
    csv_label_output_path - path to the output csv file, storing labels for the entire dataset
    json_curr_label_path - path to the .json file, containing videos and labels for them
    
    NOTE:
        output csv file for storing dataset labels have 
        the same fields as json label files. 
        The purpose of the method is to collect label files
        from different file sources and merge them into single file
    """
    curr_labels = pandas.read_json(path=json_curr_label_path)
    output_labels = pandas.read_csv(csv_label_output_path)
    output_labels = pandas.concat([output_labels, curr_labels], axis=0)
    output_labels.to_csv(path_or_buf=csv_label_output_path)

def move_video(output_dir: str, source_dir: str, video_id: str, ext: str):
    """
    Moves video file from one 
    folder to another.
    
    Parameters:
    -----------
    output_dir - destination path of the video
    source_dir - source path of the video
    video_id - video hash (unique)
    ext - extension of the video file
    """
    output_path = os.path.join(output_dir, video_id + "." + ext)
    source_path = os.path.join(source_dir, video_id + "." + ext)
    shutil.move(src=source_path, dst=output_path)

def split_videos(
    json_curr_label_path: str, 
    slice_videos_dir: str, 
    output_orig_dir: str, 
    output_fake_dir: str
):
    """
    Splits dataset videos 
    into fake and original ones, moving
    them to separate folders, dedicated for their corresponding label
    
    Parameters:
    -----------
    json_curr_label_path: str - json file, containing labels for videos to offload
    slice_videos_dir - directory, containing videos to process
    output_orig_dir - path, where to store original videos
    output_fake_dir - path, where to store deep faked videos
    """
    labels = pandas.read_json(json_curr_label_path)
    for video_id in labels.index.unique().tolist():

        # processing fake video

        if labels.iloc[video_id]['label'].str == "FAKE":
            original_video_id = labels.iloc[video_id]['label'].str
            move_video(
                output_dir=output_fake_dir, 
                source_dir=slice_videos_dir, 
                video_id=video_id,
                ext=os.path.splitext(video_id)[1]
            )

        else:
            original_video_id = video_id
        
        # saving original video

        move_video(
            output_dir=output_orig_dir, 
            source_dir=slice_videos_dir, 
            video_id=original_video_id,
            ext=os.path.splitext(video_id)[1]
        )

def main():

    parser = argparse.ArgumentParser(description="CLI Pipeline for decomposing dataset slices")
    arg = parser.add_argument 

    arg("--output-labels-path", type=str, dest='output_labels_path', help='path, where output labels are going to be stored')
    arg("--dataset-slice-dir", type=str, dest='dataset_slice_dir', help='directory, where dataset slice is stored')
    arg("--dataset-slice-labels-path", type=str, dest='dataset_slice_labels_path', help='path to the dataset slice labels file')
    arg("--output-orig-dir", type=str, dest='output_orig_dir', help='path for storing original videos')
    arg("--output-fake-dir", type=str, dest='output_fake_dir', help='path for storing deep faked videos')
    
    args = parser.parse_args()

    # initializing parameters

    dataset_videos_dir = pathlib.Path(args.dataset_slice_dir)
    slice_labels_path = pathlib.Path(args.dataset_slice_labels_path)
    output_labels_path = pathlib.Path(args.output_labels_path)
    output_orig_dir = pathlib.Path(args.output_orig_dir)
    output_fake_dir = pathlib.Path(args.output_fake_dir)

    # recreating directories, in case they are not initialized

    os.makedirs(output_labels_path)
    os.makedirs(output_orig_dir)
    os.makedirs(output_fake_dir)

    # offloading labels into main labels file
    merge_labels(
        csv_label_output_path=output_labels_path, 
        json_curr_label_path=slice_labels_path
    )

    # split videos into fake and original directories
    split_videos(
        json_curr_label_path=slice_labels_path,
        slice_videos_dir=dataset_videos_dir,
        output_orig_dir=output_orig_dir,
        output_fake_dir=output_fake_dir
    )

if __name__  == "__main__":
    main()



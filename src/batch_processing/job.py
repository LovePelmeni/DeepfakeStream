import os
import logging
import torch
import pandas
import pathlib
import glob
import cv2
from src.preprocessing.crop_augmentations import get_inference_augmentations
from src.inference import predict
import logging


Logger = logging.getLogger(name='job_logger')
handler = logging.FileHandler(filename="job_logs.log")
Logger.addHandler(handler)

def job_pipeline():
    try:
        DATA_SOURCE_PATH = os.environ.get("DATA_SOURCE_PATH")
        DATA_SAVE_PATH = os.environ.get("DATA_SAVE_PATH")
        DATA_SAVE_FORMAT = os.environ.get("DATA_SAVE_FORMAT")
        DATA_BATCH_ID = os.environ.get("DATA_BATCH_ID")
        INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE")
        CONFIG_PATH = os.environ.get("CONFIG_PATH")
    except(Exception) as err:
        Logger.error(err)
        Logger.error("Some of the critical environment variables haven't been found for running a job.")

    extensions = os.environ.get("FILE_EXTENSIONS")
    available_formats = ["pkl", "json", "csv"]
    if DATA_SAVE_FORMAT.lower() not in available_formats:
        Logger.error("invalid saving format '%s', available formats: %s" % 
        (
            DATA_SAVE_FORMAT,
            available_formats
        ))
        raise SystemExit("")

    images = []
    save_dir = pathlib.Path(DATA_SAVE_PATH)
    os.makedirs(save_dir, exist_ok=True)

    for ext in extensions.split(","):
        for img_path in glob.glob(DATA_SOURCE_PATH, pathname="*/*.%s" % ext):
            full_img_path = os.path.join(DATA_SOURCE_PATH, img_path)
            try:
                image_file = torch.as_tensor(
                    cv2.imread(full_img_path, cv2.IMREAD_UNCHANGED)
                ).permute(2, 0, 1)
                images.append(image_file)
            except(Exception) as img_err:
                
                Logger.error(
                "Failed to load image under path: %s, ERROR: %s" % (
                    full_img_path,
                    img_err
                ))

    inference_model = predict.InferenceModel.from_config(config_path=CONFIG_PATH)

    for image in images:
        predicted_args = inference_model.predict(input_img=image)
        row = pandas.Series(data=predicted_args, index=list(predicted_args.keys()))
        output_df = pandas.concat([output_df, row], axis=0)

    # saving output data
    full_save_path = os.path.join(DATA_SAVE_PATH, DATA_BATCH_ID + ".%s" %  DATA_SAVE_FORMAT)

    if DATA_SAVE_FORMAT.lower() == "csv":
        output_df.to_csv(path_or_buf=full_save_path, index=False)

    elif DATA_SAVE_FORMAT.lower() == "json":
        output_df.to_json(path_or_buf=full_save_path, index=False)

    elif DATA_SAVE_FORMAT.lower() == "pkl":
        output_df.to_pickle(path=full_save_path)

    Logger.debug("batch has been successfully processed and saved.")

if __name__ == "__main__":
    job_pipeline()
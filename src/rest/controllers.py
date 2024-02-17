import logging
import fastapi.responses 
import fastapi.exceptions 

from src.inference import predict
from fastapi import UploadFile, File
from PIL import Image 
from io import BytesIO
import os
import numpy

from src.monitoring import (
    server_monitoring
)

logger = logging.getLogger("controller_logger")
file_handler = logging.FileHandler(filename="controller_logger.log")
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler.setFormatter(formatter)
logger.addHandler(logger)

# Loading inference model pipeline from configuration file

try:
    config_path = os.environ.get("INFERENCE_CONFIG_PATH")
    model = predict.InferenceModel.from_config(config_path=config_path)

except(FileNotFoundError) as err:
    raise SystemExit("Failed to load model inference configuration, check logs.")

async def predict_human_deepfake(image_file: UploadFile = File(...)):
    """
    Accepts image, containing human faces and predicts the probability
    of each human face, being a deepfake
    
    Parameters:
    ----------
        - "image_hash" - base64 encoded string, which represents image
    """
    try:
        img_content = await image_file.read()
        img_arr = numpy.asarray(Image.open(fp=BytesIO(img_content)))
        
        # predicting deepfakes 
        predictions = model.predict(input_img=img_arr)

        # closing file after reading
        await image_file.close()

        return fastapi.responses.JSONResponse(
            status_code=201, 
            content={
                'predictions': predictions
            }
        )

    except(fastapi.exceptions.HTTPException) as val_err:
        logger.error(val_err)
        return fastapi.responses.JSONResponse(
            status_code=400, 
            content={'error': 'Prediction Failed'}
        )

async def healthcheck():
    """
    Returns 200OK if server is up and running,
    otherwise bad code.
    """
    return fastapi.responses.Response(status_code=200)

async def parse_system_metrics():
    """
    Returns set of information about
    health of the overall system.
    """
    metrics_content = server_monitoring.parse_server_info()
    return fastapi.responses.Response(
        status_code=200,
        content=metrics_content
    )
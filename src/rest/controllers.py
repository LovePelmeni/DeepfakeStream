import logging
from PIL import Image
import fastapi.responses 
import fastapi.exceptions 
from fastapi import Request
from src.inference import predict

import os
import numpy
import base64

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

async def predict_human_deepfake(request: Request):
    """
    Controller for predicting human deepfake,
    based on the showed photo
    """
    try:
        img_bytes_string = (await request.form()).get("image_hash")

        # decodes base64 string -> load to PIL Image object -> converts to numpy representation
        img = numpy.asarray(
            Image.open(
                fp=base64.b64decode(s=img_bytes_string)
            )
        )
        # predicting deepfakes 
        predictions = model.predict(input_img=img)
        
        return fastapi.responses.Response(
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
    return fastapi.responses.Response(status_code=200)
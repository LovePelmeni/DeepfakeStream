import logging
from PIL import Image
from fastapi import (
    responses,
    exceptions
)
from fastapi import Request
from src.inference import predict

import os
import numpy
import base64

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
    config_path = os.environ.get("INFERENCE_CONFIG_PATH", None)
    print(config_path)
    if config_path is None:
        raise SystemExit("""
        Inference config not found. 
        Set 'INFERENCE_CONFIG_PATH' environment variable. 
        Specify path to configuration file, containing inference configuration.""")

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
        
        return responses.Response(
            status_code=201, 
            content={
                'predictions': predictions
            }
        )

    except(exceptions.HTTPException) as val_err:
        logger.error(val_err)
        return responses.JSONResponse(
            status_code=400, 
            content={'error': 'Prediction Failed'}
        )

async def healthcheck():
    return responses.Response(status_code=200)

async def parse_system_metrics():
    
    metrics_content = server_monitoring.parse_server_info()
    return responses.Response(
        status_code=200,
        content=metrics_content
    )
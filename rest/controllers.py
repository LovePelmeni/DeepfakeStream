from fastapi import UploadFile 
import logging
from PIL import Image
import torch
import fastapi.responses 
import fastapi.exceptions 

logger = logging.getLogger("controller_logger")
file_handler = logging.FileHandler(filename="controller_logger.log")
logger.addHandler(logger)

try:
    model = torch.jit.load("models/inf_deepfake_model.onnx")
except(FileNotFoundError) as err:
    raise SystemExit("Failed to load model file, check logs.")


async def predict_human_deepfake(photo: UploadFile = ...):
    """
    Controller for predicting human deepfake,
    based on the showed photo
    """
    try:
        img = [Image.open(photo)]
        predicted_label = model.forward(img)
        return fastapi.responses.Response(
            status_code=201, content={
                "label": predicted_label
            }
        )

    except(fastapi.exceptions.HTTPException) as val_err:
        logger.error("model prediction failed")
        logger.error(val_err)
        return fastapi.responses.JSONResponse(
            status_code=400, 
            content={'error': 'Prediction Failed'}
        )

async def healthcheck():
    return fastapi.responses.Response(status_code=200)
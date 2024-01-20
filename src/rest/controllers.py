import logging
from PIL import Image

import torch
import fastapi.responses 
import fastapi.exceptions 
from fastapi import Request

import os
import numpy
import base64

GPU_DEVICE = "cuda"

logger = logging.getLogger("controller_logger")
file_handler = logging.FileHandler(filename="controller_logger.log")
logger.addHandler(logger)


network = None
try:
    network = torch.load("models/inf_deepfake_model.onnx")
except(FileNotFoundError) as err:
    # raise SystemExit("Failed to load model file, check logs.")
    pass


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
        tensor_img = torch.from_numpy(img).to(GPU_DEVICE)
        predicted_labels = network.forward(tensor_img).cpu()
        output_label = torch.argmax(predicted_labels, dim=0)
        
        return fastapi.responses.Response(
            status_code=201, content={
                "label": output_label
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